import subprocess
import os

def encoding_ladder(input_file):
    
    # Define encoding ladder configurations
    encoding_configs = [
        {"scale": "1920:1080", "bitrate_video": "5000k", "output_suffix": "1080p"},
        {"scale": "1280:720", "bitrate_video": "2500k", "output_suffix": "720p"},
        {"scale": "854:480", "bitrate_video": "1000k", "output_suffix": "480p"},
    ]

    output_files = {}

    for config in encoding_configs:
        output_file = f"{os.path.splitext(input_file)[0]}_{config['output_suffix']}.mp4"
        ffmpeg_command = [
            "ffmpeg", "-i", input_file,
            "-vf", f"scale={config['scale']}",
            "-b:v", config["bitrate_video"],
            "-c:v", "libx264", "-preset", "faster",
            "-c:a", "aac", "-b:a", "128k",
            "-f", "mp4", output_file
        ]

        try:
            print(f"Encoding {config['output_suffix']}...")
            subprocess.run(ffmpeg_command, check=True)
            output_files[config["output_suffix"]] = output_file
        except subprocess.CalledProcessError as e:
            print(f"Error while encoding {config['output_suffix']}: {e}")
            output_files[config["output_suffix"]] = None

    return output_files

