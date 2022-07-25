#!/usr/bin/env python
# coding: utf-8

"""
Synchronises Electrophysiology Spikes to Video.
"""

import numpy as np

from scipy.signal import argrelmax
from scipy.io.wavfile import write

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import mat73

from moviepy.editor import VideoClip, VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip, CompositeAudioClip, clips_array
from moviepy.video.io.bindings import mplfig_to_npimage

class SpikeVideo:
    def __init__(self, ephys_file, video_file) -> None:
        """SpikeVideo class constructor.

        Parameters
        ----------
        ephys_file : str
            Electrophysiology recordings file location.
        video_file : str
            Corresponding video file location.
        """
        self.ephys_file = ephys_file
        self.video_file = video_file

        self.data_dict = mat73.loadmat(self.ephys_file)["ephys"]
        # trace       = filtered data (artifact replaced with nans)
        # times       = times of the recording (in sec)
        # stim_on     = stimuli onset (in sec)
        # stim_off    = stimuli offset (in sec)
        # pupil_times = the time vector of the video (in sec)

        self.video = VideoFileClip(self.video_file)

        self.spike_sampling_freq = np.argwhere(self.data_dict["times"] >= 1)[0][0]
        self.video_frame_rate = int(np.ceil(self.video.fps))

        spike_threshold = 0.0005
        peak_indices = np.squeeze(argrelmax(self.data_dict["trace"]))
        self.spike_times = self.data_dict["times"][peak_indices[self.data_dict["trace"][peak_indices] > spike_threshold]]


    def plot_spikes(self, t_b, t_a=0, ax=None, plot_type='spike', shade_stimulus=True, indicate_spikes=False):
        """Plot the spike waveform.

        Parameters
        ----------
        t_b : float
            Window start time (s).
        t_a : float, optional
            Window start time (s), by default 0.0.
        ax : matplotlib.axes.Axes, optional
            Axis to plot on, by default None.
        plot_type : str, optional
            Plot type, by default 'spike'.
            'spike' - Spike time plot.
            'raster' - Spike rater plot.
            'histogram' - PSTH plot.
        shade_stimulus : bool, optional
            Shade the stimulus area for 'spike' plot, by default False.
        indicate_spikes : bool, optional
            Indicate spikes with a vertical line for 'spike' plot, by default False.

        Returns
        -------
        numpy.ndarray
            X-axis data.
        numpy.ndarray
            Y-axis data.
        list of matplotlib objects
            Value returned by specified plotting function, to be used for animation.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(16, 4), dpi=40)

        a = int(t_a * self.spike_sampling_freq)
        n = int((t_b - t_a) * self.spike_sampling_freq) # Window size

        if plot_type == 'spike':
            ax.axhline(0, color='k', linestyle='--', linewidth=1)

            if shade_stimulus:
                for son, soff in zip(self.data_dict["stim_on"], self.data_dict["stim_off"]):
                    if son*self.spike_sampling_freq >= a+n: break
                    soff = min(soff, (a+n)/self.spike_sampling_freq)
                    ax.axvspan(son, soff, color='pink', alpha=0.1)

            if indicate_spikes:
                for spike_time in self.spike_times[(self.spike_times >= t_a) & (self.spike_times < t_b)]:
                    ax.axvline(spike_time, color='g', linestyle='-.')
        
            return self.data_dict["times"][a:a+n], self.data_dict["trace"][a:a+n], ax.plot(self.data_dict["times"][a:a+n], self.data_dict["trace"][a:a+n])
        elif plot_type == 'raster':
            return
        elif plot_type == 'histogram':
            return
        
        ## FFT
        # ax[1].axhline(0, color='k', linestyle='--')
        # ax[1].plot(np.fft.fftshift(np.fft.fftfreq(n, 1/self.spike_sampling_freq)), np.abs(np.fft.fftshift(np.fft.fft(self.data_dict["trace"][a:a+n]))))
        # ax[1].set_xticks(np.arange(-int(self.spike_sampling_freq/2), int(self.spike_sampling_freq/2), 1000))
        # ax[1].set_xlim([-self.spike_sampling_freq/2, self.spike_sampling_freq/2]);

    def generate_plot(self, t_b, t_a=0.0, speed=1.0, t_w=1.0, fps=None, plot_type='spike', transition='roll', save=True, output=".temp/__temp__.mp4"):
        """Generate plot animation of specified plot type.

        Parameters
        ----------
        t_b : float
            Window start time (s).
        t_a : float, optional
            Window start time (s), by default 0.0.
        speed : float, optional
            Playback speed, by default 1.0.
        fps : int, optional
            Frames per second, by default None.
        t_w : float, optional
            Window size (s) if transition is 'window', by default 1.0.
        plot_type: str, optional
            Plot type, by default 'spike'.
            'spike' - Spike time plot.
            'raster' - Spike rater plot.
            'histogram' - PSTH plot.
        transition : {'roll', 'window'}, optional
            Type of video animation, by default 'roll'.
            'roll' - Growing spike waveform.
            'window' - Shifting window waveform.
        save : bool, optional
            Save video as .mp4?, by default True.
        output : str, optional
            Output file name, by default '.temp/__temp__.mp4'.

        Returns
        -------
        matplotlib.animation.FuncAnimation or moviepy.editor.VideoClip object
            Video object file for layouting.
        int
            FPS of the video.
        """
        if speed is None:
            if transition == 'roll':
                speed = round((t_b - t_a)/5.0, 3) #1-5
            elif transition == 'window':
                speed = round(t_w/1.0, 3)         #1-5

        sfps = max(24, np.ceil(fps if fps else min(120, self.spike_sampling_freq*speed, self.video_frame_rate*speed)))

        if transition == 'roll': t_w = 0
        print(f"> Estimated Video Length ~ {(t_b - t_w - t_a)/speed:.2f}s @ {sfps:>5} fps  | [{t_a:.3f}s, {t_b:.3f}s] @ {speed}x <{transition[0]}> -- `{output if output else f'{transition}.mp4'}`")

        fig, ax = plt.subplots(1, 1, figsize=(16, 4), dpi=40)
        ax.set_ylim(-0.001, 0.002)
        ax.axis('off')

        if transition == 'window' and plot_type == 'spike':
            def make_frame(t):
                ax.clear()
                ax.set_xlim(t_a + t*speed, t_a + t_w + t*speed)        
                self.plot_spikes(t_a + t_w + t*speed, t_a + t*speed, ax=ax, plot_type='spike', shade_stimulus=True, indicate_spikes=False)
                return mplfig_to_npimage(fig)
            
            video = VideoClip(make_frame, duration=(t_b - t_w - t_a)/speed)
            if save: video.write_videofile(output if output else f"{transition}.mp4", fps=sfps, verbose=False, logger=None)
            # video.ipython_display(fps=sfps, loop=False, autoplay=True)
            
        elif transition == 'roll':
            ax.set_xlim(t_a, t_b)

            x, y, (line,) = self.plot_spikes(t_b, t_a, ax=ax, plot_type=plot_type, shade_stimulus=True, indicate_spikes=False)
            plt.tight_layout()

            def update(frame):
                f = int(frame*self.spike_sampling_freq*speed/sfps)
                line.set_data(x[:f], y[:f])
                return line,

            video = animation.FuncAnimation(fig, update, frames=int((t_b - t_a)/speed*sfps), interval=int(1000/sfps), blit=True)
            if save: video.save(output if output else f"{transition}.mp4", writer=animation.FFMpegWriter(fps=sfps))
        
        plt.close()
        
        return video, sfps

    def generate_audio(self, t_b, t_a=0.0, speed=1.0, tone_frequency=200, tone_duration=0.002, save=True, output='.temp/__temp__.wav'):
        """ Generate spike audio.

        Parameters
        ----------
        t_b : float
            Window start time (s).
        t_a : float, optional
            Window start time (s), by default 0.0.
        speed : float, optional
            Playback speed, by default 1.0.
        tone_frequency : int, optional
            Frequency of spike chirp, by default 440.
        tone_duration : float, optional
            Duration of spike chirp, by default 0.002.
        save : bool, optional
            Save audio as .wav?, by default True.
        output : str, optional
            Output file name, by default '.temp/__temp__.wav'.

        Returns
        -------
        numpy.ndarray
            Audio waveform.
        int
            Samples/s of the audio signal.
        """
        if speed is None: speed = round((t_b - t_a)/5.0, 3) #1-5
        
        afps = int(self.spike_sampling_freq*speed)
        if tone_frequency >= afps/2:
            raise ValueError(f"Speed too low! Min Speed - {2*tone_frequency/self.spike_sampling_freq}")
        
        print(f"> Estimated Audio Length ~ {(t_b - t_a)/speed:.2f}s @ {afps:>5} afps | [{t_a:.3f}s, {t_b:.3f}s] @ {speed}x     -- `{output}`")
        
        n = int((t_b - t_a) * self.spike_sampling_freq)
        #x = np.arange(n)*(t_b - t_a)/(n-1)/speed
        
        def add_chirp(signal, index):
            if index + int(self.spike_sampling_freq*tone_duration) < signal.size:
                signal[index:index+int(self.spike_sampling_freq*tone_duration)] = np.sin(2*np.pi*tone_frequency/speed*np.linspace(0, tone_duration, int(self.spike_sampling_freq*tone_duration)))
        
        chirp_indices = ((self.spike_times[(self.spike_times >= t_a) & (self.spike_times < t_b)] - t_a) * self.spike_sampling_freq).astype(np.int)

        y = np.zeros(n)
        for chirp_index in chirp_indices:
            add_chirp(y, chirp_index)

        audio = np.array(y/max(y), dtype=np.float32)
        if save: write(output, afps, audio)
        
        return audio, afps

    def generate(self, t_b, t_a=0.0, speed=1.0, output="spike.mp4"):
        """Generate complete video.

        Parameters
        ----------
        t_b : float
            Window start time (s).
        t_a : float, optional
            Window start time (s), by default 0.0.
        speed : float, optional
            Playback speed, by default 1.0.
        output : str, optional
            Output file name, by default '.temp/__temp__.wav'.
        """
        _, afps = self.generate_audio(t_b, t_a, speed, save=True, output=".temp/__temp__.wav")
        _, sfps = self.generate_plot(t_b, t_a, speed, save=True, output=".temp/__temp__.mp4")
        
        spike_video = VideoFileClip(".temp/__temp__.mp4")
        spike_audio = AudioFileClip(".temp/__temp__.wav")
        spike_video.audio = CompositeAudioClip([spike_audio])
        video_x = self.video.subclip(t_a, t_b).speedx(speed).set_fps(sfps)
        
        txt_clip = TextClip(f"x{speed}", fontsize=75, color='black')
        txt_clip = txt_clip.set_position(("right", "top")).set_duration((t_b-t_a)/speed)
        
        clips_array([[CompositeVideoClip([video_x, txt_clip])], [spike_video]]).write_videofile(output) #, verbose=False, logger=None)

if __name__ == '__main__':
    pass
