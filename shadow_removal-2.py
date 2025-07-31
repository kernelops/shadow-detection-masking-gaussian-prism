import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops
from skimage.color import rgb2hsv
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def create_pipeline_visualization(images, titles, grid_shape=(2, 3)):
    """
    Creates a flexible dashboard with a grid layout (e.g., 2x3).
    """
    rows, cols = grid_shape
    num_images = len(images)

    # Determine a standard size based on the first image
    h, w = images[0].shape[:2]
    img_size = (w, h)

    # Create a dark gray canvas
    padding = 10
    title_height = 40
    canvas_h = rows * (h + title_height + padding) + padding
    canvas_w = cols * (w + padding) + padding
    dashboard = np.full((canvas_h, canvas_w, 3), (48, 48, 48), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    for i, (img, title) in enumerate(zip(images, titles)):
        if i >= rows * cols: break

        # Prepare the image
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        resized_img = cv2.resize(img, img_size)

        # Calculate position
        row_idx = i // cols
        col_idx = i % cols
        x_start = padding + col_idx * (w + padding)
        y_start = padding + row_idx * (h + title_height + padding)

        # Place the image and its title
        dashboard[y_start + title_height : y_start + title_height + h, x_start : x_start + w] = resized_img
        cv2.putText(dashboard, title, (x_start, y_start + title_height - 10), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
    return dashboard

def _rgb_to_ycbcr(rgb_image):
    T = np.array([[0.299,0.587,0.114],[-0.168736,-0.331264,0.5],[0.5,-0.418688,-0.081312]])
    ycbcr = np.dot(rgb_image, T.T); ycbcr[:, :, 1:] += 128
    return ycbcr

def _histogram_matching(source, template):
    shape = source.shape; source = source.ravel(); template = template.ravel()
    s_vals, s_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_vals, t_counts = np.unique(template, return_counts=True)
    s_quant = np.cumsum(s_counts).astype(float)/s_counts.sum()
    t_quant = np.cumsum(t_counts).astype(float)/t_counts.sum()
    interp_vals = np.interp(s_quant, t_quant, t_vals)
    return interp_vals[s_idx].reshape(shape)

class ShadowRemover:
    def __init__(self, image_path, n_segments=400, compactness=30):
        self.original_image_bgr = cv2.imread(image_path)
        if self.original_image_bgr is None: raise FileNotFoundError(f"Image not found: {image_path}")
        if self.original_image_bgr.ndim == 2: self.original_image_bgr = cv2.cvtColor(self.original_image_bgr, cv2.COLOR_GRAY2BGR)
        self.image_rgb = cv2.cvtColor(self.original_image_bgr, cv2.COLOR_BGR2RGB)
        self.image_float = self.image_rgb / 255.0
        self.height, self.width, _ = self.image_float.shape
        self.n_segments = n_segments; self.compactness = compactness
        self.shadow_labels = None
        self.disparity_matrix = None
        print("ShadowRemover initialized.")

    def run_pipeline(self):
        """Executes the entire shadow detection and removal pipeline."""
        self._segment_image()
        self._calculate_features_and_disparity()
        self._detect_shadows_iteratively()
        shadow_mask = self._generate_final_mask()
        removed_result = self._remove_shadows_paper_method(shadow_mask)
        return shadow_mask, removed_result

    def _segment_image(self):
        print("1. Segmenting image..."); 
        self.segments = slic(self.image_float, n_segments=self.n_segments, compactness=self.compactness, start_label=1)
        self.region_props = regionprops(self.segments)
        self.num_regions = len(self.region_props)

    def _calculate_features_and_disparity(self):
        print("2. Calculating features and region disparity matrix...")
        gray = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2GRAY)
        hsv = rgb2hsv(self.image_float)
        ycbcr = _rgb_to_ycbcr(self.image_float)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0); grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        self.region_features = []
        for prop in self.region_props:
            mask = self.segments == prop.label
            grad_hist, _ = np.histogram(grad_mag[mask], 16, (0, 255))
            lbp_hist, _ = np.histogram(lbp[mask], 10, (0, 10))
            self.region_features.append({'grad_hist': grad_hist/(grad_hist.sum()+1e-6), 'lbp_hist': lbp_hist/(lbp_hist.sum()+1e-6), 'Y': np.mean(ycbcr[mask,0])})
        self.disparity_matrix = np.full((self.num_regions, self.num_regions), float('inf'))
        dist_norm = np.sqrt(self.height**2 + self.width**2)
        for i in tqdm(range(self.num_regions), desc="   Calculating Disparity"):
            for j in range(i + 1, self.num_regions):
                d_g=np.linalg.norm(self.region_features[i]['grad_hist']-self.region_features[j]['grad_hist'])
                d_t=np.linalg.norm(self.region_features[i]['lbp_hist']-self.region_features[j]['lbp_hist'])
                d_d=np.linalg.norm(np.array(self.region_props[i].centroid)-np.array(self.region_props[j].centroid))/dist_norm
                self.disparity_matrix[i,j] = self.disparity_matrix[j,i] = d_g + d_t + d_d

    def _detect_shadows_iteratively(self):
        print("3. Iteratively detecting shadows...")
        y_feats = np.array([f['Y'] for f in self.region_features])
        self.shadow_labels = np.full(self.num_regions, 255) # 0=shadow, 1=lit, 255=undecided
        self.shadow_labels[y_feats < 0.5 * y_feats.mean()] = 0
        shadow_indices = np.where(self.shadow_labels == 0)[0]
        for i in shadow_indices:
            best_match_j = np.argmin(self.disparity_matrix[i])
            if self.shadow_labels[best_match_j] != 0: self.shadow_labels[best_match_j] = 1
        print("Shadow detection logic complete.")

    def _generate_final_mask(self):
        mask = np.zeros(self.image_rgb.shape[:2], dtype="uint8")
        for i, label in enumerate(self.shadow_labels):
            if label == 0: mask[self.segments == self.region_props[i].label] = 255
        return mask

    def _remove_shadows_paper_method(self, shadow_mask):
        print("4. Removing shadows using paper's relighting method...")
        hsv_image = rgb2hsv(self.image_float); relit_hsv = hsv_image.copy()
        for i in tqdm(range(self.num_regions), desc="   Relighting Regions"):
            if self.shadow_labels[i] == 0:
                candidate_disparities = self.disparity_matrix[i].copy()
                candidate_disparities[self.shadow_labels == 0] = float('inf')
                if not np.all(np.isinf(candidate_disparities)):
                    target_idx = np.argmin(candidate_disparities)
                    s_mask = (self.segments == self.region_props[i].label)
                    t_mask = (self.segments == self.region_props[target_idx].label)
                    if np.any(s_mask) and np.any(t_mask):
                        for ch in range(3): relit_hsv[s_mask,ch] = _histogram_matching(hsv_image[s_mask,ch], hsv_image[t_mask,ch])
        print("5. Smoothing boundaries...")
        relit_rgb_f32 = cv2.cvtColor(relit_hsv.astype(np.float32), cv2.COLOR_HSV2RGB)
        boundary_mask = cv2.dilate(shadow_mask,np.ones((5,5),np.uint8),2) - cv2.erode(shadow_mask,np.ones((5,5),np.uint8),2)
        blurred_img = cv2.GaussianBlur(relit_rgb_f32, (15,15), 0)
        final_float = np.where(boundary_mask[:,:,None].astype(bool), blurred_img, relit_rgb_f32)
        final_image = (np.clip(final_float, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)

if __name__ == '__main__':
    input_image_path = "img3.png"  # <--- INPUT IMAGE TO REMOVE SHADOW
    
    try:
        remover = ShadowRemover(input_image_path, n_segments=400, compactness=40)
        shadow_mask, removed_result = remover.run_pipeline()
        
        segmentation_viz = mark_boundaries(remover.image_float, remover.segments, color=(1,1,0))
        segmentation_viz_bgr = cv2.cvtColor((segmentation_viz * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        matching_viz = segmentation_viz_bgr.copy()
        for i in range(remover.num_regions):
            if remover.shadow_labels[i] == 0:
                candidate_disparities = remover.disparity_matrix[i].copy()
                candidate_disparities[remover.shadow_labels == 0] = float('inf')
                if not np.all(np.isinf(candidate_disparities)):
                    j = np.argmin(candidate_disparities)
                    y1, x1 = map(int, remover.region_props[i].centroid)
                    y2, x2 = map(int, remover.region_props[j].centroid)
                    cv2.line(matching_viz, (x1,y1), (x2,y2), (255,0,0), 1)

        images_to_display = [
            remover.original_image_bgr,
            segmentation_viz_bgr,
            matching_viz,
            shadow_mask,
            removed_result
        ]
        titles = [
            "1. Input Image",
            "2. Segmentation",
            "3. Region Matching",
            "4. Shadow Detection",
            "5. Shadow Removal"
        ]

        visualization = create_pipeline_visualization(images_to_display, titles)
        
        print("\nDisplaying final pipeline visualization. Press any key to exit.")
        cv2.imshow("Shadow Removal Pipeline (Paper's Method)", visualization)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please make sure the 'input_image_path' variable is set correctly.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
