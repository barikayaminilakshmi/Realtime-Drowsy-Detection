import mediapipe as mp
print("mediapipe file:", mp.__file__)
print("has solutions:", hasattr(mp, "solutions"))
print("version:", getattr(mp, "__version__", "no __version__"))