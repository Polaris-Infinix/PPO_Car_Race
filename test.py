import numpy as np 
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # Output: (2, 
import gymnasium as gym
import torch
import cv2
import numpy as np
env = gym.make("CarRacing-v3",render_mode="human", lap_complete_percent=1, domain_randomize=True, continuous=True)
obs, info = env.reset()
i=0
# Example: create a dummy RGB image tensor (C, H, W)
while True:
    print(env.action_space.sample())
    arr=np.array([0,0,0])
    obs, reward, done, truncated, info = env.step(arr)
    print(reward)
    
    image_tensor = torch.from_numpy(obs)  # Random image in CHW format

    # Convert it to (H, W, C) for display
    image_np = image_tensor.permute(2,0,1).numpy()
    i=i+1

    image_bgr = (image_np[1] * 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)

    # Step 3: Show the image using OpenCV
    cv2.imshow("Tensor Image", image_bgr)
    cv2.waitKey(0)  # Wait for a key press to close
    cv2.destroyAllWindows()
    print(i)

# Dry run for first 50 frames 
