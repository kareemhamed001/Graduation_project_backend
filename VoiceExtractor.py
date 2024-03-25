from moviepy.editor import VideoFileClip


def extract_audio(video_file, audio_file):
    # Load the video file
    video_clip = VideoFileClip(video_file)

    # Extract audio
    audio_clip = video_clip.audio

    # Write audio to WAV file
    audio_clip.write_audiofile(audio_file, codec='pcm_s16le')

    # Close the video clip
    video_clip.close()


# Replace 'input_video.mp4' and 'output_audio.wav' with your file paths
input_video = 'E:/python/4rt grade first term/Deepfake_detection_video/static/uploads/0c99c094-5165-475b-9151-c8a9b3854937.mp4'
output_audio = 'output_audio.wav'

extract_audio(input_video, output_audio)

