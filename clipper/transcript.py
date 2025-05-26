import os
import whisper


class Transcript:

    @staticmethod
    def get_whisper(input_video_path: str,
                    whisper_model: str = 'medium',
                    word_timestamps: bool = True):
        """
        Получение транскрипции с помощью Whisper.
        Возвращает подробные таймкоды слов, если установлен флаг word_timestamps.
        """
        if not os.path.isfile(input_video_path):
            raise FileNotFoundError("Файл с видео не найден.")

        audio_data = whisper.load_audio(input_video_path)
        model = whisper.load_model(whisper_model)

        transcribe_args = {
            "fp16": False,
            "task": "transcribe"
        }

        if word_timestamps:
            transcribe_args["word_timestamps"] = True
            transcribe_args["beam_size"] = 5  # требуется для точных таймингов слов

        result = whisper.transcribe(model, audio_data, **transcribe_args)
        return result

    @staticmethod
    def get_text(transcript_input):
        # Поддержка как raw списка, так и словаря от whisper с ключом "segments"
        segments = transcript_input.get("segments") if isinstance(transcript_input, dict) else transcript_input

        output_text = ""
        for segment in segments:
            start = segment.get("start", 0.0)
            end = segment.get("end", start + segment.get("duration", 0))
            text = segment.get("text", "")
            output_text += f"{start:.2f} --> {end:.2f} : {text}\n"

        return output_text
