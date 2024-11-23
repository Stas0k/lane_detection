import cv2
import matplotlib.pyplot as plt
import math
from moviepy.editor import VideoFileClip, ImageSequenceClip
from IPython.display import HTML
import re


class BaseUtils(object):
    write_to_file = False
    
    @classmethod
    def plot_images(cls, *img_info, width=5, height=5, **kwargs):
        """
        Plots images
        :param img_info: list of dicts. Each dict represent a separate image with it's attributes (e.g. image title)
        Example: {'img': <image numpy array>, 'title': 'image-1'}
        :param width: subplot width (default: 5)
        :param height: subplot heigh (default: 5)
        :param kwargs: additional keyword arguments
        """
        num_imgs = len(img_info)
        n_cols = min(num_imgs, kwargs.get('n_cols', 4))
        n_rows = math.ceil(num_imgs / n_cols)

        fig = plt.figure(figsize=(n_cols * width, n_rows * height))
        for i, ii in enumerate(img_info):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ii_img =  ii.get('img')
            ii_cmap = ii.get('cmap')
            ii_title = ii.get('title')
            ax.imshow(ii_img, cmap=ii_cmap)
            ax.set_title(ii_title)
        
            axis = kwargs.get('axis')
            if axis is not None:
                plt.axis(axis)

        plt.tight_layout()

        save_fig = kwargs.get('save_fig')
        if save_fig and cls.write_to_file:
            # Save the plot to an image file
            plt.savefig(save_fig, bbox_inches ="tight")
        
        plt.show()

    @classmethod
    def save_frame(cls, video_path, frame_number, output_image_path):
        """
        Saves specific frame from the video to image file
        Usage example:
            save_frame('path_to_video.mp4', 100, 'output_image.jpg')
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Check if the video opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read the frame
        ret, frame = cap.read()
        
        if ret:
            # Save the frame as an image file
            cv2.imwrite(output_image_path, frame)
            print(f"Frame {frame_number} saved as {output_image_path}")
        else:
            print(f"Error: Could not read frame {frame_number}")
        
        # Release the video capture object
        cap.release()

    @classmethod
    def display_image_with_coordinates(cls, img, cvt_color=cv2.COLOR_RGB2BGR):
        img_cp = img.copy()
        if cvt_color:
            img_cp = cv2.cvtColor(img_cp, cvt_color)
        # Function to display coordinates on mouse click
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"Coordinates: X={x}, Y={y}")
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_cp, f"{x},{y}", (x,y), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.imshow('image', img_cp)

        # Check if the image was loaded successfully
        if img_cp is None:
            print("Error: Could not load image.")
            return

        # Display the image
        cv2.imshow('image', img_cp)

        # Set the mouse callback function to display coordinates
        cv2.setMouseCallback('image', click_event)

        # Wait for a key press and close the image window
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class VidProc(object):
    write_to_file = False

    @classmethod
    def resize_video(cls, input_path, output_path, width, height):
        # Open the input video
        cap = cv2.VideoCapture(input_path)
        
        # Get the original video's width, height, and frames per second (fps)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Define the codec and create VideoWriter object to save the resized video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize the frame
            resized_frame = cv2.resize(frame, (width, height))
            
            # Write the resized frame to the output video
            out.write(resized_frame)
        
        # Release everything when done
        cap.release()
        out.release()

    @classmethod
    def display_video(cls, video, method='window', time_range=None, apply_func=None, video_size='960x540', title=None, **kwargs):
        if method == 'window':
            if type(video) == VideoFileClip:
                v_clip = video if time_range is None else video.subclip(*time_range)
                # Display the transformed video using OpenCV
                for frame in v_clip.iter_frames(fps=video.fps, dtype="uint8"):
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
                    cv2.imshow('Transformed Video', frame)
                    if cv2.waitKey(int(1000 / v_clip.fps)) & 0xFF == ord('q'):
                        break

                cv2.destroyAllWindows()
            elif isinstance(video, str):
                cap = cv2.VideoCapture(video)
                while (cap.isOpened()):
                    ret, frame = cap.read() # Cap.read() returns a ret bool to indicate success.
                    # If frame is read correctly, ret is True
                    if not ret:
                        print("Can't receive frame or stream was ended.")
                        break

                    cv2.imshow('Lane Detection', apply_func(frame))
                    if cv2.waitKey(int(1000 / video.fps)) & 0xFF == ord('q'):
                        break

                print("Releasing camera and closing windows")
                cap.release() # camera release 
                cv2.destroyAllWindows()          
        elif method == 'html':
            width, height = video_size.split('x')
            margin_right = kwargs.get('margin_right', '10px')
            div_width =  kwargs.get('div_width', '100%')
            
            if not isinstance(video, (tuple, list)):
                video = [video]
            
            if not isinstance(title, (tuple, list)):
                title = [title]

            video_html = '<div style="display: flex; justify-content: space-between; margin-left: auto; margin-right: auto; width: {dw}">'
            for i, (vid_path, vid_title) in enumerate(zip(video, title)):
                video_html += """
                <div style="text-align: center; margin: {mr};">
                    <h3>{t}</h3>
                    <video width="{w}" height="{h}" controls>
                        <source src="{v}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>
                """.format(v=vid_path, w=width, h=height, mr=margin_right, dw=div_width, t=vid_title if vid_title else f"video-{i}")
            video_html += '</div>'
            return HTML(video_html)
            # return \
            # HTML("""
            # <video width="960" height="540" controls>
            # <source src="{0}" type="video/mp4">
            # </video>
            # """.format(video))

    
    @classmethod
    def write_video(cls, video_clip, output_file, audio=False):
        if cls.write_to_file:
            video_clip.write_videofile(output_file, audio=audio)

    @classmethod
    def get_clip_frame(cls, video_clip, frame_indx):
        t, input_type = None, None
        if isinstance(frame_indx, (int, float)):
            t = frame_indx
            input_type = 'index'
        elif isinstance(frame_indx, str):
            if frame_indx.isdigit():
                t = int(frame_indx)
                input_type = 'index'
            else:
                # find the number followed by 'sec'
                what_matched = re.search(r'(\d+)sec', frame_indx)
                if what_matched:
                    t = int(what_matched.group(1))
                    input_type = "time"
        #print(f"get_clip_frame=> input:{frame_indx}, res:{t * 1.0 / video_clip.fps}")
        return video_clip.get_frame(t if input_type == 'time' else t * 1.0 / video_clip.fps)
    
    @classmethod
    def video_clip_to_gif(cls, video_clip, gif_path, num_frames):

        # Calculate the duration of the video and the interval between frames
        duration = video_clip.duration
        interval = duration / num_frames
        
        # Extract frames at equal intervals
        frames = [video_clip.get_frame(i * interval) for i in range(num_frames)]
        
        # Create a new clip with the extracted frames
        gif_clip = ImageSequenceClip(frames, fps=video_clip.fps)
        
        # Write the GIF to the specified path
        gif_clip.write_gif(gif_path)

    