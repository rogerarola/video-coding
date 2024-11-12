# Use the official Python 3.9 image as a base
FROM python:3.9

# Set the working directory
WORKDIR /code

# Copy the requirements file and install dependencies
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the main application file
COPY main.py /code/main.py

<<<<<<< Updated upstream
# Run the FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
=======
COPY /app /code/app


CMD ["fastapi", "run", "app/main.py", "--port", "80"]

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app


>>>>>>> Stashed changes
