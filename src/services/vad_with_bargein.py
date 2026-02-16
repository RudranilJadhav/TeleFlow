import numpy as np
import torch
from silero_vad import load_silero_vad
from collections import deque
import time
from typing import Optional

class BargeInConfig: 

    def __init__(self):
        # VAD thresholds – low to catch quiet speech
        self.vad_threshold = 0.3          
        self.speech_threshold = 0.4       

        # Energy thresholds – only to filter out pure silence or clipping
        self.min_energy = 1e-4            
        self.max_energy = 0.5              

        # Timing parameters (ms)
        self.min_speech_for_bargein = 200    
        self.min_utterance_ms = 250           
        self.silence_timeout_ms = 500         
        self.barge_in_cooldown_ms = 800       
        self.hangover_ms = 200                 

        # Smoothing
        self.vad_smoothing_window = 3          

        # Adaptive noise floor 
        self.noise_floor_alpha = 0.995       

        # Backchannel words
        self.backchannel_words = {
            "yes", "yeah", "yep", "ok", "okay", "uh", "um", "ah", "hmm",
            "mm", "mhm", "uhh", "right", "sure", "no", "nope", "hey",
            "hi", "hello", "thanks", "thank you", "good", "great"
        }

class VADWithBargeIn:

    def __init__(self, config: Optional[BargeInConfig] = None, sample_rate: int = 16000):
        
        self.config = config if config is not None else BargeInConfig()
        
        self.sample_rate = sample_rate
        
        self.vad_model = load_silero_vad().to("cpu").eval()
        self.vad_frame_ms = 32
        self.vad_samples = int(sample_rate * self.vad_frame_ms / 1000)
        
        self.min_speech_frames = max(1, self.config.min_speech_for_bargein // self.vad_frame_ms)
        self.silence_timeout_frames = self.config.silence_timeout_ms // self.vad_frame_ms
        self.barge_in_cooldown_frames = self.config.barge_in_cooldown_ms // self.vad_frame_ms
        self.hangover_frames = self.config.hangover_ms // self.vad_frame_ms
        
        self.vad_buffer = deque(maxlen=self.config.vad_smoothing_window)

        self.noise_floor = 1e-4
        self.noise_floor_alpha = self.config.noise_floor_alpha

        self.vad_history = deque(maxlen=self.min_speech_frames * 2)
        self.energy_history = deque(maxlen=self.min_speech_frames * 2)

        self.speech_frames = 0
        self.silence_frames = 0
        self.hangover_counter = 0
        self.cooldown_frames = 0
        self.last_barge_in_time = 0

        self.utterance_buffer = np.array([], dtype=np.float32)
        self.is_speaking = False 

    def process_frame(self, audio_chunk: np.ndarray) -> dict:
        result = {
            'vad_prob': 0.0,
            'is_speech': False,         
            'should_barge_in': False,
            'utterance_complete': False,
            'utterance_audio': None,
        }

        # Ensure chunk is 32ms
        if len(audio_chunk) != self.vad_samples:
            if len(audio_chunk) < self.vad_samples:
                audio_chunk = np.pad(audio_chunk, (0, self.vad_samples - len(audio_chunk)))
            else:
                audio_chunk = audio_chunk[:self.vad_samples]

        # Get Silero VAD probability
        with torch.no_grad():
            prob = self.vad_model(
                torch.from_numpy(audio_chunk),
                self.sample_rate
            ).item()
        result['vad_prob'] = prob

        # Smooth probability
        self.vad_buffer.append(prob)
        smoothed_prob = np.mean(self.vad_buffer) if self.vad_buffer else prob

        # Compute RMS energy
        rms = np.sqrt(np.mean(audio_chunk**2))

        # Update noise floor during silence (when prob is low and RMS not too high)
        if smoothed_prob < 0.2 and rms < self.config.max_energy:
            self.noise_floor = self.noise_floor_alpha * self.noise_floor + (1 - self.noise_floor_alpha) * rms

        # Determine if this frame contains speech
        is_speech_frame = (smoothed_prob > self.config.speech_threshold and
                           rms > self.noise_floor * 2.0 and   
                           rms < self.config.max_energy)

        result['is_speech'] = is_speech_frame

        # Update cooldown
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1

        # Track history for barge-in analysis
        self.vad_history.append(smoothed_prob)
        self.energy_history.append(rms)

        # --- State machine ---
        if is_speech_frame:
            # Speech present
            self.speech_frames += 1
            self.silence_frames = 0
            self.hangover_counter = self.hangover_frames   # Reset hangover

            # Add to utterance buffer
            self.utterance_buffer = np.concatenate([self.utterance_buffer, audio_chunk])

            # Check for barge-in (only if not already speaking)
            if (not self.is_speaking and
                self.cooldown_frames == 0 and
                self.speech_frames >= self.min_speech_frames):

                # Additional check: recent energy variation
                if self._is_likely_speech():
                    result['should_barge_in'] = True
                    self.is_speaking = True
                    self.last_barge_in_time = time.time()
                    self.cooldown_frames = self.barge_in_cooldown_frames
                    self.speech_frames = 0   # Reset after triggering
        else:
            # No speech in this frame
            if self.hangover_counter > 0:
                # Hangover period – treat as speech for buffering
                self.hangover_counter -= 1
                self.utterance_buffer = np.concatenate([self.utterance_buffer, audio_chunk])
            else:
                # Real silence
                self.silence_frames += 1

                if self.is_speaking:
                    # Check for end of utterance
                    if self.silence_frames >= self.silence_timeout_frames and len(self.utterance_buffer) > 0:
                        result['utterance_complete'] = True
                        result['utterance_audio'] = self.utterance_buffer.copy()
                        # Reset
                        self.utterance_buffer = np.array([], dtype=np.float32)
                        self.is_speaking = False
                        self.speech_frames = 0
                else:
                    # Not speaking – after a short silence, reset speech counter
                    if self.silence_frames > 5:   # ~160ms
                        self.speech_frames = 0

        return result

    def _is_likely_speech(self) -> bool:

        if len(self.energy_history) < self.min_speech_frames:
            return True   # Not enough history, assume speech

        # Compute variance of recent energy
        recent_energy = list(self.energy_history)[-self.min_speech_frames:]
        energy_std = np.std(recent_energy)
        energy_mean = np.mean(recent_energy)

        # If energy is too constant (low variance), might be noise
        if energy_mean > 0 and (energy_std / energy_mean) < 0.02:
            return False   # Too constant – probably noise
        return True

    def is_backchannel(self, text: str) -> bool:
        
        if not text:
            return True
        text = text.lower().strip()
        # Single word
        if text in self.config.backchannel_words:
            return True
        # Short phrases (1-2 words)
        words = text.split()
        if len(words) <= 2 and all(w in self.config.backchannel_words for w in words):
            return True
        return False

    def reset(self):
        
        self.vad_buffer.clear()
        self.vad_history.clear()
        self.energy_history.clear()
        self.speech_frames = 0
        self.silence_frames = 0
        self.hangover_counter = 0
        self.cooldown_frames = 0
        self.utterance_buffer = np.array([], dtype=np.float32)
        self.is_speaking = False
        self.noise_floor = 1e-4

def create_vad_with_bargein(sample_rate=16000):
    return VADWithBargeIn(sample_rate=sample_rate)