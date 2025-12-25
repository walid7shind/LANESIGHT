import yt_dlp

def download_youtube_chunk(url, start_time, end_time, output_filename="video_chunk.mp4"):
    """
    Download a specific chunk of a YouTube video.
    
    Args:
        url: YouTube video URL
        start_time: Start time in format "HH:MM:SS" or "MM:SS"
        end_time: End time in format "HH:MM:SS" or "MM:SS"
        output_filename: Name of the output file
    """
    ydl_opts = {
        'format': 'best',
        'outtmpl': output_filename,
        'download_ranges': yt_dlp.utils.download_range_func(None, [(start_time, end_time)]),
        'force_keyframes_at_cuts': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    print(f"Downloaded chunk from {start_time} to {end_time} successfully!")

if __name__ == "__main__":
    video_url = "https://youtu.be/ZOZOqbK86t0?si=q5XGMadTJJIStPum"
    start = "4:27:55"
    end = "4:30:00"
    
    download_youtube_chunk(video_url, start, end, "video_chunk_4_27_55_to_4_30_00.mp4")

