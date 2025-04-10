import cv2
import os

def extract_frames(video_path, output_folder):
    # 출력 폴더가 없으면 생성
    os.makedirs(output_folder, exist_ok=True)
    
    # 동영상 열기
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 프레임을 더 이상 읽을 수 없으면 종료

        # 프레임을 PNG로 저장
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:06d}.png")
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1

    cap.release()
    print(f"추출된 프레임 수: {frame_count}")

# 사용 예시
print(os.getcwd())
video_path = "./example/sample_video/2025_03_18_08.12.17.mp4"
output_folder = "./example/sample_video/frames"
extract_frames(video_path, output_folder)
