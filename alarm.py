import pygame

def play_sound(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

if __name__ == "__main__":
    sound_file_path = "path/to/your/sound/file.mp3"  # Replace with the actual path to your sound file
    play_sound(sound_file_path)