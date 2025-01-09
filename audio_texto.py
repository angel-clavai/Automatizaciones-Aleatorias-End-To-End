import speech_recognition as sr

def transcribe_audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        
    try:
        #reconocimiento de Google como backend de Speech-to-Text
        text = recognizer.recognize_google(audio_data, language="es-ES")
        return text
    except sr.UnknownValueError:
        print("No se pudo entender el audio")
        return ""
    except sr.RequestError as e:
        print(f"Error con el servicio de Google Speech Recognition: {e}")
        return ""



if __name__ == '__main__':
    audio_file = "path_to_your_audio_file.wav"
    transcribed_text = transcribe_audio_to_text(audio_file)
    print("Texto transcrito:", transcribed_text)
