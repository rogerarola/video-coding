from fastapi import FastAPI, UploadFile, Form
import os
import subprocess

app = FastAPI()

#folder for video uploads
UPLOAD_FOLDER = "./app/videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#1
@app.post("/modify-resolution/")
async def modify_resolution(file: UploadFile, resolution: str = Form(...)):
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_path = os.path.join(UPLOAD_FOLDER, f"output_{resolution}_{file.filename}")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    #change resolution using FFmpeg
    subprocess.run([
        "ffmpeg", "-i", input_path, "-s", resolution, "-c:a", "copy", output_path
    ])

    #clean up input file
    os.remove(input_path)
    return {"message": f"Resolution modified. Saved as {output_path}"}

#2
@app.post("/modify-chroma/")
async def modify_chroma(file: UploadFile, pix_fmt: str = Form(...)):
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_path = os.path.join(UPLOAD_FOLDER, f"output_{pix_fmt}_{file.filename}")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    #modify chroma subsampling using FFmpeg
    subprocess.run([
        "ffmpeg", "-i", input_path, "-pix_fmt", pix_fmt, output_path
    ])

    os.remove(input_path)
    return {"message": f"Chroma subsampling modified. Saved as {output_path}"}

#3
@app.post("/video-info/")
async def video_info(file: UploadFile):
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(input_path, "wb") as f:
        f.write(await file.read())

    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", 
         "format=duration:stream=codec_name,codec_type,width,height", 
         "-of", "json", input_path],
        capture_output=True,
        text=True
    )

    os.remove(input_path)
    return {"video_info": result.stdout}

#4
@app.post("/create-container/")
async def create_container(file: UploadFile):
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_folder = UPLOAD_FOLDER
    output_path = os.path.join(output_folder, "output_bbb.mp4")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    #audio and video processing with FFmpeg
    subprocess.run([
        "ffmpeg", "-i", input_path, "-t", "20", "-c:v", "libx264", 
        "-c:a", "aac", "-ac", "1", os.path.join(output_folder, "output_aac.m4a")
    ])
    subprocess.run([
        "ffmpeg", "-i", input_path, "-t", "20", "-c:v", "libx264", 
        "-c:a", "mp3", "-ac", "2", "-b:a", "128k", os.path.join(output_folder, "output_mp3.mp3")
    ])
    subprocess.run([
        "ffmpeg", "-i", input_path, "-t", "20", "-c:v", "libx264", 
        "-c:a", "ac3", os.path.join(output_folder, "output_ac3.ac3")
    ])
    subprocess.run([
        "ffmpeg", "-i", input_path, "-t", "20", "-map", "0:v", "-map", "0:a", 
        "-c:v", "libx264", "-c:a", "aac", output_path
    ])

    os.remove(input_path)
    return {"message": f"BBB container created at {output_path}"}

#5
@app.post("/count-tracks/")
async def count_tracks(file: UploadFile):
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(input_path, "wb") as f:
        f.write(await file.read())

    #track information using FFprobe
    result = subprocess.run(
        ["ffprobe", "-i", input_path, "-show_entries", "stream=index", "-select_streams", "v,a,s", "-of", "json"],
        capture_output=True,
        text=True
    )

    os.remove(input_path)

    #JSON output to count tracks
    import json
    output = json.loads(result.stdout)
    num_tracks = len(output.get("streams", []))

    return {"num_tracks": num_tracks, "message": f"The file contains {num_tracks} track(s)."}

#6
@app.post("/macroblocks/")
async def visualize_macroblocks(file: UploadFile):
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_path = os.path.join(UPLOAD_FOLDER, f"macroblocks_{file.filename}")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    #macroblock visualization using FFMPEG
    subprocess.run([
        "ffmpeg", "-i", input_path, "-vf", "codecview=mv=pf+bf+bb", "-an", output_path
    ])

    os.remove(input_path)
    return {"message": f"Macroblock visualization saved as {output_path}"}

#7
@app.post("/yuv-histogram/")
async def yuv_histogram(file: UploadFile):
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_path = os.path.join(UPLOAD_FOLDER, f"yuv_histogram_{file.filename}")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    #YUV histogram using FFMPEG
    subprocess.run([
        "ffmpeg", "-i", input_path, "-vf", "split=2[a][b],[b]histogram,format=yuv420p[v]", "-map", "[a]", "-map", "[v]",
        output_path
    ])

    os.remove(input_path)
    return {"message": f"YUV histogram saved as {output_path}"}
