import cv2
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    model = tf.keras.models.load_model('models/unet_model-256-v1.h5')
    bg = cv2.imread('backgrounds/swirl.jpeg')
    bg = cv2.resize(bg, (1920, 1080))

    cap = cv2.VideoCapture(0) 

    result = cv2.VideoWriter('plots/sample_video.avi',  
                            cv2.VideoWriter_fourcc(*'MJPG'), 
                            10, (1920, 1080))

    while True:
        ret, frame = cap.read()

        # Flip Horizontal
        frame = cv2.flip(frame, 1)

        # Get frame shape
        height, width, _ = frame.shape

        # Process for prediction
        resized_frame = cv2.resize(frame, (256, 256))
        normalized_frame = resized_frame / 255.0
        input_batch = np.expand_dims(normalized_frame, axis=0)

        # Predict mask
        mask = model.predict(input_batch)[0]

        # Resize and process masks
        mask = cv2.resize(mask, (width, height))
        mask = np.expand_dims(mask, axis=-1)

        person = mask > 0.5
        bg_mask = mask <= 0.5

        person = cv2.cvtColor(person.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        bg_mask = cv2.cvtColor(bg_mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # Create new frame with bg replacement
        segmented_frame = (frame * person) + (bg * bg_mask)

        # display image
        cv2.imshow("Segmented Frame", segmented_frame)

        # Save to video
        result.write(segmented_frame) 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
