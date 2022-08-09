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

## CONSTANTS
T_W = 0                     # Window length for windowed spike plot (s)
T_W_ = (-10e-2, 10e-2)      # Window length for histogram and raster plots (s)
BIN_SIZE = 2e-3             # Bin size for histogram and raster plots (s)

TONE_FREQUENCY = 200        # Chirp frequency for audio (Hz)
TONE_DURATION = 0.002       # Chirp duration for audio (s)

FIG_SIZE = {
    'spike': (16, 3.2),
    'raster': (3.2, 6.4),
    'histogram': (3.2, 6.4),
}

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

    # def raster_spikes(self, t_b, t_a=0.0, t_w=T_W_):
    #     """Create spike raster data

    #     Parameters
    #     ----------
    #     t_b : float
    #         Window start time (s).
    #     t_a : float, optional
    #         Window start time (s), by default 0.0.
    #     t_w : tuple of floats, optional
    #         Raster time window ((s), (s)), by default `T_W_``.

    #     Returns
    #     -------
    #     numpy.ndarray
    #         Array of raster spike events (N,); N = Number of trials in the given time window; <row> = Spike times relative to the stimulus of that trial.
    #     """
    #     spike_times = self.spike_times[(self.spike_times >= t_a) & (self.spike_times < t_b)]
    #     stimulus_times = self.data_dict["stim_on"][(self.data_dict["stim_on"] >= t_a) & (self.data_dict["stim_on"] < t_b)]

    #     time_indices = np.array([((spike_times >= stimulus_time + t_w[0]) & (spike_times < stimulus_time + t_w[1])).argmax() for stimulus_time in stimulus_times])
        
    #     for k, j in enumerate(time_indices):
    #         if k and not j and time_indices[k-1]:
    #             time_indices[k] = max(time_indices[k:]) and min(time_indices[k:][np.nonzero(time_indices[k:])]) or -1

    #     trials = (np.array(np.split(spike_times, time_indices)[1:]).T - stimulus_times).T
    #     for j, trial in enumerate(trials):
    #         trials[j] = trial[(trial >= t_w[0]) & (trial < t_w[1])]
            
    #     return np.around(trials, 6)

    def plot_spikes(self, t_b, t_a=0.0, ax=None, t_w=T_W_, bin_size=BIN_SIZE, plot_type='spike', figsize=None, shade_stimulus=True, indicate_spikes=False):
        """Plot the spike waveform.

        Parameters
        ----------
        t_b : float
            Window start time (s).
        t_a : float, optional
            Window start time (s), by default 0.0.
        ax : matplotlib.axes.Axes, optional
            Axis to plot on, by default None.
        t_w : tuple of floats, optional
            PSTH and raster plot limits (s), if `plot_type` is 'raster' or 'histogram', by default `T_W_`.
        bin_size : float, optional
            PSTH binsize (s), if `plot_type` is 'histogram', by default `BIN_SIZE`.
        plot_type : str, optional
            Plot type, by default 'spike'.
            'spike' - Spike time plot.
            'raster' - Spike raster plot.
            'histogram' - PSTH plot.
        figsize : tuple, optional
            Size of figure, by default None.
            Reads from `FIG_SIZE` based on `plot_type` if None
        shade_stimulus : bool, optional
            Shade the stimulus area for 'spike' plot, by default False.
        indicate_spikes : bool, optional
            Indicate spikes with a vertical line for 'spike' plot, by default False.

        Returns
        -------
        list of matplotlib objects
            Value returned by specified plotting function, to be used for animation.
        numpy.ndarray
            X-axis data.
        numpy.ndarray
            Y-axis data.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(figsize if figsize else FIG_SIZE[plot_type]), dpi=40)

        a = int(t_a * self.spike_sampling_freq)
        n = int((t_b - t_a) * self.spike_sampling_freq) # Window size

        if plot_type == 'spike':
            x = self.data_dict["times"][a:a+n]
            y = self.data_dict["trace"][a:a+n]

            ax.axhline(0, color='r', linestyle='--', linewidth=1)

            if shade_stimulus:
                for son, soff in zip(self.data_dict["stim_on"], self.data_dict["stim_off"]):
                    if son*self.spike_sampling_freq >= a+n: break
                    soff = min(soff, (a+n)/self.spike_sampling_freq)
                    ax.axvspan(son, soff, color='pink', alpha=0.1)

            if indicate_spikes:
                for spike_time in self.spike_times[(self.spike_times >= t_a) & (self.spike_times < t_b)]:
                    ax.axvline(spike_time, color='r', linestyle='-.')
        
            obj, = ax.plot(x, y, color='k')
        else:
            x = spike_times = self.spike_times[(self.spike_times >= t_a) & (self.spike_times < t_b)]
            y = stimulus_times = self.data_dict["stim_on"][(self.data_dict["stim_on"] >= t_a) & (self.data_dict["stim_on"] < t_b)]

            # trials = self.rater_spikes(t_b, t_a, t_w=t_w)

            if plot_type == 'raster':
                n_trials = len(stimulus_times)
                obj = ax.eventplot([[]]*n_trials, color='k')
            elif plot_type == 'histogram':
                n_bins = int((t_w[1] - t_w[0])/bin_size)
                *_, obj = ax.hist([], bins=n_bins, range=t_w, color='k')
                # *_, obj = ax.hist(np.around(np.concatenate(trials), 5), bins=n_bins, range=t_w)[0]/len(trials)#/bin_size
                # bin_centers = np.linspace(t_w[0] + bin_size/2, t_w[1] - bin_size/2, n_bins)
        
        return obj, x, y
        
        ## FFT
        # ax[1].axhline(0, color='r', linestyle='--')
        # ax[1].plot(np.fft.fftshift(np.fft.fftfreq(n, 1/self.spike_sampling_freq)), np.abs(np.fft.fftshift(np.fft.fft(self.data_dict["trace"][a:a+n]))), color='k')
        # ax[1].set_xticks(np.arange(-int(self.spike_sampling_freq/2), int(self.spike_sampling_freq/2), 1000))
        # ax[1].set_xlim([-self.spike_sampling_freq/2, self.spike_sampling_freq/2]);

    def generate_plot(self, t_b, t_a=0.0, speed=1.0, t_w=None, bin_size=BIN_SIZE, fps=None, plot_type='spike', transition='roll', figsize=None, save=True, output=".temp/__temp__.mp4"):
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
        t_w : float or tuple of floats, optional
            Window size (s), if transition is 'window', by default `T_W`.
            PSTH and raster plot limits ((s), (s)), if `plot_type` is 'raster' or 'histogram', by default `T_W_`.
        bin_size : float, optional
            PSTH binsize (s), if `plot_type` is 'histogram', by default `BIN_SIZE`.
        plot_type : str, optional
            Plot type, by default 'spike'.
            'spike' - Spike time plot.
            'raster' - Spike raster plot.
            'histogram' - PSTH plot.
        transition : {'roll', 'window'}, optional
            Type of video animation, by default 'roll'.
            'roll' - Growing spike waveform.
            'window' - Shifting window waveform.
        figsize : tuple, optional
            Size of figure, by default None.
            Reads from `FIG_SIZE` based on `plot_type` if None
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

        if t_w is None:
            if plot_type in ('raster', 'histogram'):
                t_w = T_W_
            elif transition == 'window': 
                t_w = T_W

        if transition == 'window' and plot_type != 'spike': transition = 'roll'
        if type(t_w) in (list, tuple) and transition == 'window': t_w = t_w[1] - t_w[0]
        elif type(t_w) in (float, int) and plot_type in ('raster', 'histogram'): t_w = (-t_w, t_w)

        print(f"> Estimated Video Length ~ {(t_b - (0 if transition == 'roll' else t_w) - t_a)/speed:.2f}s @ {sfps:>5} fps  | [{t_a:.3f}s, {t_b:.3f}s] @ {speed}x <{transition[0]}> -- `{output}`")

        fig, ax = plt.subplots(1, 1, figsize=(figsize if figsize else FIG_SIZE[plot_type]), dpi=40)

        if plot_type == 'spike':
            if transition == 'roll':
                ax.set_xlim(t_a, t_b)
            else:
                ax.set_ylim(-0.001, 0.002)
            ax.axis('off')
        else:
            if plot_type == 'raster':
                ax.set_xticks([])

            ax.set_xlim(t_w)
            ax.set_yticks([])
            ax.axvline(0, color='r', linestyle='--')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            # ax.spines['bottom'].set_visible(False)

        if transition == 'window' and plot_type == 'spike':
            def make_frame(t):
                ax.clear()
                ax.set_xlim(t_a + t*speed, t_a + t_w + t*speed)        
                self.plot_spikes(t_a + t_w + t*speed, t_a + t*speed, ax=ax, plot_type='spike', shade_stimulus=True, indicate_spikes=False)
                return mplfig_to_npimage(fig)
            
            video = VideoClip(make_frame, duration=(t_b - t_w - t_a)/speed)
            if save: video.write_videofile(output, fps=sfps, verbose=False, logger=None)
            # video.ipython_display(fps=sfps, loop=False, autoplay=True)
            
        elif transition == 'roll':
            obj, *xy = self.plot_spikes(t_b, t_a, ax=ax, t_w=t_w, bin_size=bin_size, plot_type=plot_type, shade_stimulus=True, indicate_spikes=False)
            plt.tight_layout()
            spike_times, stimulus_times = xy
            
            if plot_type == 'raster':
                ax.set_ylim([-0.5, len(obj) - 0.5])
            elif plot_type == 'histogram':
                ax.set_ylim([0, len(obj)/10])

            def update(frame):
                s = frame*speed/sfps
                f = int(s*self.spike_sampling_freq)
                if plot_type == 'spike':
                    obj.set_data(xy[0][:f], xy[1][:f])
                    return obj,
                else:
                    spike_times_i = spike_times[np.argwhere((spike_times - t_a >= s - speed/sfps) & (spike_times - t_a < s))].flatten()
                    if plot_type == 'raster':
                        for tn, (wa, wb) in enumerate(zip(stimulus_times - t_a + t_w[0], stimulus_times - t_a + t_w[1])):
                            [obj[-tn-1].add_positions(st - wa + t_w[0]) for st in spike_times_i - t_a if (st >= wa) and (st < wb)]
                        return obj
                    elif plot_type == 'histogram':
                        counts = [0]*len(obj)
                        # np.histogram([], bins=len(obj), range=t_w)
                        for wa, wb in zip(stimulus_times - t_a + t_w[0], stimulus_times - t_a + t_w[1]):
                            for st in spike_times_i - t_a:
                                if (st >= wa) and (st < wb):
                                    counts[int((st - wa)*len(obj)/(t_w[1] - t_w[0]))] += 1
                        for count, rect in zip(counts, obj.patches):
                            rect.set_height(rect.get_height() + count)
                        return obj.patches

            video = animation.FuncAnimation(fig, update, frames=int((t_b - t_a)/speed*sfps), interval=int(1000/sfps), blit=False)
            if save: video.save(output, writer=animation.FFMpegWriter(fps=sfps))
        
        plt.close()
        
        return video, sfps

    def generate_audio(self, t_b, t_a=0.0, speed=1.0, tone_frequency=TONE_FREQUENCY, tone_duration=TONE_DURATION, save=True, output=".temp/__temp__.wav"):
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
            Frequency of spike chirp, by default `TONE_FREQUENCY`.
        tone_duration : float, optional
            Duration of spike chirp, by default `TONE_DURATION`.
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

    def generate_video(self, t_b, t_a=0.0, speed=1.0, t_w=T_W_, bin_size=BIN_SIZE, tone_frequency=TONE_FREQUENCY, tone_duration=TONE_DURATION, output=".temp/__output__.mp4"):
        """Generate complete video.

        Parameters
        ----------
        t_b : float
            Window start time (s).
        t_a : float, optional
            Window start time (s), by default 0.0.
        speed : float, optional
            Playback speed, by default 1.0.
        t_w : tuple of floats, optional
            PSTH and raster plot limits ((s), (s)), if `plot_type` is 'raster' or 'histogram', by default `T_W_`.
        bin_size : float, optional
            PSTH binsize (s), if `plot_type` is 'histogram', by default `BIN_SIZE`.
        tone_frequency : int, optional
            Frequency of spike chirp, by default `TONE_FREQUENCY`.
        tone_duration : float, optional
            Duration of spike chirp, by default `TONE_DURATION`.
        output : str, optional
            Output file name, by default '.temp/__output__.mp4'.
        """
        _, sfps = self.generate_plot(t_b, t_a, speed, t_w, bin_size, save=True, output=".temp/__temp__spike__.mp4")
        _______ = self.generate_plot(t_b, t_a, speed, t_w, bin_size, save=True, plot_type='raster', output=".temp/__temp__raster__.mp4")
        _______ = self.generate_plot(t_b, t_a, speed, t_w, bin_size, save=True, plot_type='histogram', output=".temp/__temp__histogram__.mp4")
        
        _, afps = self.generate_audio(t_b, t_a, speed, tone_frequency, tone_duration, save=True, output=".temp/__temp__.wav")

        video_x = self.video.subclip(t_a, t_b).speedx(speed).set_fps(sfps)

        spike_video = VideoFileClip(".temp/__temp__spike__.mp4")
        raster_video = VideoFileClip(".temp/__temp__raster__.mp4")
        histogram_video = VideoFileClip(".temp/__temp__histogram__.mp4")

        spike_audio = AudioFileClip(".temp/__temp__.wav")
        spike_video.audio = CompositeAudioClip([spike_audio])

        txt_clip = TextClip(f"x{speed}", fontsize=75, color='black')
        txt_clip = txt_clip.set_position(("right", "top")).set_duration((t_b-t_a)/speed)
        
        clips_array([[CompositeVideoClip([video_x, txt_clip]), raster_video], [spike_video, histogram_video]]).write_videofile(output) #, verbose=False, logger=None)

if __name__ == '__main__':
    pass
