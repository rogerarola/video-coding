import streamlit as st
import sys
from pathlib import Path
import os
import shutil

# Add the app directory to the system path to ensure Python can find the modules
sys.path.append(str(Path(__file__).resolve().parent))

from services.video_converter import video_converter
from services.encoding_ladder import encoding_ladder

# Custom CSS for green background and styles
st.markdown(
    """
    <style>
    .main { background-color: #e8f5e9; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎥 Transcend: The Ultimate Video Converter 🚀")

# File upload
uploaded_file = st.file_uploader("Upload your video file", type=["mp4", "mkv", "avi", "webm"])

if uploaded_file:
    # Save the uploaded file to a temporary location
    input_path = f"/tmp/{uploaded_file.name}"
    with open(input_path, "wb") as buffer:
        buffer.write(uploaded_file.read())
    st.success(f"✅ File uploaded successfully: {uploaded_file.name}")

    # Menu selection
    menu = st.radio("What would you like to do?", ["Convert Video", "Generate Encoding Ladder"])

    if menu == "Convert Video":
        # Conversion section
        format_option = st.selectbox("Select the target format", ["vp8", "vp9", "h265", "av1"])

        if st.button("Start Conversion"):
            st.info(f"Converting video to {format_option}...")
            output_message = video_converter(input_path, format_option)

            if "successfully converted" in output_message:
                # Extract the output file path from the message
                output_file = output_message.split("to ")[1].strip()
                st.success(output_message)

                # Move the file to Streamlit's temporary download directory
                download_path = f"downloads/{os.path.basename(output_file)}"
                os.makedirs("downloads", exist_ok=True)
                shutil.move(output_file, download_path)

                # Provide a Streamlit download button
                with open(download_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Converted File",
                        data=f,
                        file_name=os.path.basename(download_path),
                        mime="video/webm" if output_file.endswith(".webm") else "video/mp4"
                    )
            else:
                st.error(output_message)

    elif menu == "Generate Encoding Ladder":
        # Encoding ladder section
        if st.button("Generate Encoding Ladder"):
            st.info("Generating encoding ladder...")
            output_files = encoding_ladder(input_path)

            if output_files:
                st.success("🏆 Encoding ladder generated successfully!")
                for resolution, file_path in output_files.items():
                    if file_path:
                        # Move the file to Streamlit's temporary download directory
                        download_path = f"downloads/{os.path.basename(file_path)}"
                        os.makedirs("downloads", exist_ok=True)
                        shutil.move(file_path, download_path)

                        # Provide a Streamlit download button
                        with open(download_path, "rb") as f:
                            st.download_button(
                                label=f"📥 Download {resolution} Version",
                                data=f,
                                file_name=os.path.basename(download_path),
                                mime="video/webm" if file_path.endswith(".webm") else "video/mp4"
                            )
                    else:
                        st.error(f"❌ Failed to generate {resolution}")

    # Clean up temporary files after processing
    if st.button("Clean Up"):
        os.remove(input_path)
        st.info("Temporary files removed.")

# Footer Section
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #2e7d32;'>Developed with ❤️ using Streamlit</h5>", unsafe_allow_html=True)
