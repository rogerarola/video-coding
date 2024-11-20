import subprocess

def resize_image(input_path, output_path, width, height):
    command = ['ffmpeg', '-i', input_path, '-vf', f'scale={width}:{height}', output_path]
    subprocess.run(command, check=True)
