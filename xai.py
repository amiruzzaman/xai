import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import tensorflow as tf

# import numpy as np
# import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.models import Model
# import tensorflow as tf

# Load and preprocess the image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)  # Load image and resize
    img_array = img_to_array(img)  # Convert to array
    img_array = preprocess_input(img_array)  # Preprocess for VGG16
    return img, np.expand_dims(img_array, axis=0)  # Add batch dimension

# Grad-CAM implementation
def grad_cam(model, img_array, layer_name):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)  # Gradients of the predicted class
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Mean of gradients

    conv_outputs = conv_outputs[0]  # Remove batch dimension
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    # Normalize heatmap to a range of 0 to 1
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

    return heatmap

# Superimpose heatmap on original image
def superimpose_heatmap(img, heatmap, alpha=0.4):
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (224, 224)).numpy()
    heatmap_colored = np.uint8(255 * heatmap_resized.squeeze())
    heatmap_colored = plt.cm.jet(heatmap_colored)[:, :, :3]  # Apply colormap

    superimposed_img = heatmap_colored * alpha + img / 255.0
    return np.clip(superimposed_img, 0, 1)

# # Main function
# def main(image_path):
#     # Load pretrained VGG16 model
#     model = VGG16(weights="imagenet")
#     
#     # Load and preprocess image
#     img, img_array = load_and_preprocess_image(image_path)
#     
#     # Get Grad-CAM heatmap
#     heatmap = grad_cam(model, img_array, "block5_conv3")
#     
#     # Superimpose heatmap on the original image
#     overlay = superimpose_heatmap(img_to_array(img), heatmap)
# 
#     # Plot results
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.imshow(img)
#     plt.title("Original Image")
#     plt.axis("off")
#     
#     plt.subplot(1, 2, 2)
#     plt.imshow(overlay)
#     plt.title("Grad-CAM Heatmap")
#     plt.axis("off")
#     
#     plt.show()

# Main function
def main(image_path):
    # Load pretrained VGG16 model
    model = VGG16(weights="imagenet")
    
    # Load and preprocess image
    img, img_array = load_and_preprocess_image(image_path)
    
    # Get classification results
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # Get top 3 predictions
    
    s = []
    print("Classification Results:")
    for label, class_name, probability in decoded_predictions:
        print(f"{class_name}: {probability:.4f}")
        s.append(f"{class_name}: {probability:.4f}")
    
    # Get Grad-CAM heatmap
    heatmap = grad_cam(model, img_array, "block5_conv3")
    
    # Superimpose heatmap on the original image
    overlay = superimpose_heatmap(img_to_array(img), heatmap)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Grad-CAM Heatmap")
    #adding text inside the plot
    plt.text(45, 210, s[0], fontsize = 12, color='red',
             horizontalalignment='center',
             verticalalignment='center',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round')) #classification results
    plt.axis("off")
    
    plt.savefig('grad_cam.png') # save the figure
    plt.show()



# Example usage
image_path = "cat.jpg"  # Replace with the path to your image
main(image_path)

#pip install tensorflow matplotlib
#pip install opencv-python
