
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import face_recognition
import numpy as np
import os
import subprocess

def seleccionar_imagen():
    path = filedialog.askopenfilename(
        title="Selecciona imagen de rostro",
        filetypes=[("Imágenes", "*.jpg *.jpeg *.png")]
    )
    if path:
        entrada_imagen.set(path)

def seleccionar_video():
    path = filedialog.askopenfilename(
        title="Selecciona video",
        filetypes=[("Videos", "*.mp4 *.avi *.mov")]
    )
    if path:
        entrada_video.set(path)

def procesar():
    imagen_path = entrada_imagen.get()
    video_path = entrada_video.get()

    if not os.path.exists(imagen_path) or not os.path.exists(video_path):
        messagebox.showerror("Error", "Imagen o video no válido.")
        return

    try:
        source_image = face_recognition.load_image_file(imagen_path)
        source_locations = face_recognition.face_locations(source_image)

        if not source_locations:
            messagebox.showerror("Error", "No se detectó rostro en la imagen seleccionada.")
            return

        top, right, bottom, left = source_locations[0]
        face_crop = source_image[top:bottom, left:right]
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
        face_mask = 255 * np.ones(face_crop_rgb.shape, face_crop_rgb.dtype)

        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            messagebox.showerror("Error", "No se pudo abrir el video.")
            return

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)

        salida_sin_audio = "output.mp4"
        salida_final = "final_output.mp4"
        out = cv2.VideoWriter(salida_sin_audio, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        while True:
            ret, frame = video.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(frame_rgb)

            for face in faces:
                t, r, b, l = face
                fh, fw = b - t, r - l
                resized_face = cv2.resize(face_crop_rgb, (fw, fh))
                resized_mask = cv2.resize(face_mask, (fw, fh))
                center = (l + fw // 2, t + fh // 2)
                try:
                    frame = cv2.seamlessClone(resized_face, frame, resized_mask, center, cv2.NORMAL_CLONE)
                except:
                    pass

            out.write(frame)

        video.release()
        out.release()

        comando = [
            "ffmpeg", "-y",
            "-i", salida_sin_audio,
            "-i", video_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            salida_final
        ]

        subprocess.run(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        messagebox.showinfo("✅ Finalizado", f"Video final generado con audio:\n{salida_final}")

    except Exception as e:
        messagebox.showerror("Error inesperado", str(e))

root = tk.Tk()
root.title("Face Swap Bot con Audio")
root.geometry("640x160")
root.resizable(False, False)

entrada_imagen = tk.StringVar()
entrada_video = tk.StringVar()

tk.Label(root, text="Imagen del rostro:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
tk.Entry(root, textvariable=entrada_imagen, width=55).grid(row=0, column=1)
tk.Button(root, text="Seleccionar", command=seleccionar_imagen).grid(row=0, column=2, padx=5)

tk.Label(root, text="Video original:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
tk.Entry(root, textvariable=entrada_video, width=55).grid(row=1, column=1)
tk.Button(root, text="Seleccionar", command=seleccionar_video).grid(row=1, column=2, padx=5)

tk.Button(root, text="Iniciar procesamiento", command=procesar, bg="green", fg="white", height=2, width=30)    .grid(row=2, column=1, pady=15)

root.mainloop()
