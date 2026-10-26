import cv2

STREAM_URL = "https://192.168.1.6:4444/video/mjpeg"

cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera stream. Check the URL and Wi-Fi.")

while True:
    ok, frame = cap.read()

    if not ok:
        print("Frame unavailable")
        break

    cv2.imshow("FOD Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
