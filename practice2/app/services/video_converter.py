import ffmpeg
import os
import subprocess


def video_converter(input, format):
    # Ensure the format is in lowercase
    format = format.lower()

    # Generate the output file name
    base_name, _ = os.path.splitext(input)
    
     # Adjust container format based on codec
    if format in ["vp8", "vp9", "av1"]:
        output_file = os.path.abspath(f"{base_name}_{format}.webm")
    elif format == "h265":
        output_file = os.path.abspath(f"{base_name}_{format}.mp4")
    else:
        return "The format is not compatible."

    try:
        if format == "vp8":
            ffmpeg.input(input).output(output_file, vcodec='libvpx').run()
        elif format == "vp9":
            ffmpeg.input(input).output(output_file, vcodec='libvpx-vp9').run()
        elif format == "h265":
            ffmpeg.input(input).output(output_file, vcodec='libx265').run()
        elif format == "av1":
            ffmpeg.input(input).output(output_file, vcodec='libaom-av1', crf=30, bitrate='2M').run()               
        else:
            return "The format is not compatible."
        return f"Video successfully converted to {output_file}"
    except subprocess.CalledProcessError as e:
        return f"Error during conversion: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"
