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
    output_video = os.path.join(UPLOAD_FOLDER, "output_bbb_20s.mp4")
    aac_audio = os.path.join(UPLOAD_FOLDER, "output_aac.m4a")
    mp3_audio = os.path.join(UPLOAD_FOLDER, "output_mp3.mp3")
    ac3_audio = os.path.join(UPLOAD_FOLDER, "output_ac3.ac3")
    packaged_output = os.path.join(UPLOAD_FOLDER, "packaged_bbb.mp4")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    #cut the video to 20 seconds
    subprocess.run([
        "ffmpeg", "-i", input_path, "-t", "20", "-c:v", "libx264", "-an", output_video
    ])

    #export AAC
    subprocess.run([
        "ffmpeg", "-i", input_path, "-t", "20", "-vn", "-c:a", "aac", "-ac", "1", aac_audio
    ])

    #export MP3
    subprocess.run([
        "ffmpeg", "-i", input_path, "-t", "20", "-vn", "-c:a", "mp3", "-b:a", "128k", "-ac", "2", mp3_audio
    ])

    #export AC3
    subprocess.run([
        "ffmpeg", "-i", input_path, "-t", "20", "-vn", "-c:a", "ac3", ac3_audio
    ])

    #package all tracks into a single MP4 container
    subprocess.run([
        "ffmpeg", "-i", output_video, "-i", aac_audio, "-i", mp3_audio, "-i", ac3_audio,
        "-map", "0:v", "-map", "1:a", "-map", "2:a", "-map", "3:a",
        "-c:v", "copy", "-c:a", "copy", packaged_output
    ])

    #clean up
    os.remove(input_path)
    os.remove(output_video)
    os.remove(aac_audio)
    os.remove(mp3_audio)
    os.remove(ac3_audio)

    return {"message": f"Packaged output saved as {packaged_output}"}

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

    # YUV histogram using FFmpeg
    subprocess.run([
        "ffmpeg", "-i", input_path, "-vf", "split=2[a][b],[b]histogram=display_mode=parade:scale=log:components=YUV[v]",
        "-map", "[a]", "-map", "[v]", "-c:v", "libx264", "-preset", "fast", output_path
    ])

    #clean up
    os.remove(input_path)
    return {"message": f"YUV histogram saved as {output_path}"}

