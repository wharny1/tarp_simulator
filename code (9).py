import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

# --- Configuration Constants ---
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
TARP_SIZE = 3.0
POINTS_PER_AXIS = 17
SOLVER_ITERATIONS = 10 
GRAVITY = np.array([0.0, -9.8, 0.0])
# POLE_HEIGHT is no longer needed for the main logic but is kept for reference
POLE_HEIGHT = 1.2

# --- Tarp Class ---
class Tarp:
    def __init__(self, size, num_points):
        self.size = size
        self.num_points = num_points
        self.points = np.zeros((num_points, num_points, 3), dtype=float)
        self.old_points = np.zeros_like(self.points)
        self.pinned_points = {}

        self.springs = []
        step = size / (num_points - 1)
        for i in range(num_points):
            for j in range(num_points):
                if i < num_points - 1: self.springs.append(((i, j), (i + 1, j), step))
                if j < num_points - 1: self.springs.append(((i, j), (i, j + 1), step))
                if i < num_points - 1 and j < num_points - 1:
                    self.springs.append(((i, j), (i + 1, j + 1), np.sqrt(2) * step))
                    self.springs.append(((i + 1, j), (i, j + 1), np.sqrt(2) * step))
        self.reset()

    def reset(self):
        self.pinned_points.clear()
        step = self.size / (self.num_points - 1)
        for i in range(self.num_points):
            for j in range(self.num_points):
                self.points[i, j] = np.array([-self.size/2 + j*step, 0.01, -self.size/2 + i*step])
        self.old_points = np.copy(self.points)

    def get_tieout_indices(self):
        mid = (self.num_points - 1) // 2; end = self.num_points - 1
        return [(0,0), (0,mid), (0,end), (mid,0), (mid,mid), (mid,end), (end,0), (end,mid), (end,end)]

    def update(self, dt, dragged_point_info=None):
        for i in range(self.num_points):
            for j in range(self.num_points):
                if (i, j) not in self.pinned_points:
                    temp_pos = np.copy(self.points[i, j])
                    velocity = self.points[i, j] - self.old_points[i, j]
                    self.points[i, j] += velocity + GRAVITY * dt * dt
                    self.old_points[i, j] = temp_pos
        
        for _ in range(SOLVER_ITERATIONS):
            for p1_idx, p2_idx, rest_length in self.springs:
                p1, p2 = self.points[p1_idx], self.points[p2_idx]
                delta = p2 - p1
                dist = np.linalg.norm(delta)
                if dist > 0:
                    diff = (dist - rest_length) / dist
                    correction = delta * 0.5 * diff
                    if p1_idx not in self.pinned_points: self.points[p1_idx] += correction
                    if p2_idx not in self.pinned_points: self.points[p2_idx] -= correction
            
            if dragged_point_info:
                drag_idx = dragged_point_info['index']
                target_pos = dragged_point_info['target_pos']
                correction_vec = target_pos - self.points[drag_idx]
                self.points[drag_idx] += correction_vec * 0.8
            
            for idx, pos in self.pinned_points.items():
                self.points[idx] = pos
                
            self.points[:,:,1] = np.maximum(self.points[:,:,1], 0.0)

    def draw(self, selected_tieout_idx, dragged_tieout_idx=None):
        glColor3f(0.0, 0.4, 0.2); glBegin(GL_QUADS)
        for i in range(self.num_points-1):
            for j in range(self.num_points-1):
                glVertex3fv(self.points[i,j]); glVertex3fv(self.points[i+1,j])
                glVertex3fv(self.points[i+1,j+1]); glVertex3fv(self.points[i,j+1])
        glEnd()
        glColor3f(1.0, 1.0, 1.0)
        for i in range(self.num_points-1):
            for j in range(self.num_points-1):
                glBegin(GL_LINE_LOOP)
                glVertex3fv(self.points[i,j]); glVertex3fv(self.points[i+1,j])
                glVertex3fv(self.points[i+1,j+1]); glVertex3fv(self.points[i,j+1])
                glEnd()
        all_tieouts = self.get_tieout_indices()
        for i, tieout_idx in enumerate(all_tieouts):
            pos = self.points[tieout_idx]
            if tieout_idx == dragged_tieout_idx: glColor3f(0.0, 1.0, 1.0); glPointSize(20)
            elif i == selected_tieout_idx: glColor3f(1.0, 1.0, 0.0); glPointSize(15)
            else: glColor3f(1.0, 0.5, 0.0); glPointSize(10)
            glBegin(GL_POINTS); glVertex3fv(pos); glEnd()
            if tieout_idx in self.pinned_points:
                glColor3f(0.8,0.8,0.8); glBegin(GL_LINES)
                glVertex3fv(pos); glVertex3fv(self.pinned_points[tieout_idx]); glEnd()

# --- Helper Functions ---
def draw_ground():
    glColor3f(0.3, 0.3, 0.3); glBegin(GL_LINES)
    for i in np.arange(-5, 5.1, 0.5):
        glVertex3f(i, 0, -5); glVertex3f(i, 0, 5)
        glVertex3f(-5, 0, i); glVertex3f(5, 0, i)
    glEnd()

def draw_text(x, y, text, font):
    text_surface = font.render(text, True, (255, 255, 255), (0, 0, 0, 128))
    text_data = pygame.image.tostring(text_surface, "RGBA", True)
    glWindowPos2d(x, y)
    glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

# --- Main Application ---
def main():
    pygame.init()
    pygame.display.set_caption("3x3 Tarp Simulator - Pin Anywhere!")
    display = (WINDOW_WIDTH, WINDOW_HEIGHT)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    font = pygame.font.Font(None, 24)
    glEnable(GL_DEPTH_TEST)

    tarp = Tarp(TARP_SIZE, POINTS_PER_AXIS)
    clock = pygame.time.Clock()
    dt = 1/60.0
    
    is_ortho_view = True
    ortho_zoom, ortho_pan = 4.0, [0.0, 0.0]
    persp_dist, persp_rot = 6.0, [0, -89.9]

    selected_tieout_index = 4
    paused = False
    mouse_dragging_camera = False
    last_mouse_pos = (0, 0)
    dragged_tieout_info = None

    while True:
        modelview_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection_matrix = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); return
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: mouse_dragging_camera = True; last_mouse_pos = event.pos
                if event.button == 3:
                    mx, my = event.pos
                    closest_tieout = None; min_dist_sq = 15**2
                    for i, idx in enumerate(tarp.get_tieout_indices()):
                        screen_pos = gluProject(*tarp.points[idx], modelview_matrix, projection_matrix, viewport)
                        if screen_pos and idx not in tarp.pinned_points:
                            dist_sq = (screen_pos[0]-mx)**2 + (screen_pos[1]-(WINDOW_HEIGHT-my))**2
                            if dist_sq < min_dist_sq:
                                min_dist_sq = dist_sq; closest_tieout = {'index': idx, 'depth': screen_pos[2]}
                    if closest_tieout:
                        if dragged_tieout_info and dragged_tieout_info['index'] == closest_tieout['index']:
                            dragged_tieout_info = None
                        else: dragged_tieout_info = closest_tieout
                if is_ortho_view:
                    if event.button == 4: ortho_zoom = max(1.0, ortho_zoom-0.5)
                    elif event.button == 5: ortho_zoom = min(10.0, ortho_zoom+0.5)
                else:
                    if event.button == 4: persp_dist = max(2.0, persp_dist-0.5)
                    elif event.button == 5: persp_dist = min(20.0, persp_dist+0.5)

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1: mouse_dragging_camera = False

            if event.type == pygame.MOUSEMOTION and mouse_dragging_camera:
                dx, dy = event.pos[0]-last_mouse_pos[0], event.pos[1]-last_mouse_pos[1]
                if is_ortho_view:
                    ortho_pan[0] -= dx*0.01*(ortho_zoom/4.0); ortho_pan[1] += dy*0.01*(ortho_zoom/4.0)
                else:
                    persp_rot[0] += dx*0.2; persp_rot[1] += dy*0.2
                    persp_rot[1] = max(-89.9, min(89.9, persp_rot[1]))
                last_mouse_pos = event.pos

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_v: is_ortho_view = not is_ortho_view
                elif event.key == pygame.K_r:
                    tarp.reset(); is_ortho_view = True
                    ortho_pan, ortho_zoom = [0.0, 0.0], 4.0
                    persp_dist, persp_rot = 6.0, [0, -89.9]
                    dragged_tieout_info = None
                
                # --- CHANGE: Unified 'S' key for pinning anywhere ---
                elif event.key == pygame.K_s:
                    target_idx = None
                    if dragged_tieout_info: # Priority: pin the dragged point
                        target_idx = dragged_tieout_info['index']
                    else: # Fallback: pin the selected (yellow) point
                        target_idx = tarp.get_tieout_indices()[selected_tieout_index]

                    # Check if we have a valid target that isn't already pinned
                    if target_idx and target_idx not in tarp.pinned_points:
                        # Pin the point at its CURRENT 3D position
                        tarp.pinned_points[target_idx] = np.copy(tarp.points[target_idx])
                        
                        # If we were dragging, stop dragging
                        if dragged_tieout_info and dragged_tieout_info['index'] == target_idx:
                            dragged_tieout_info = None
                
                # --- REMOVED: The 'L' key is no longer necessary ---

                elif event.key == pygame.K_u: # Unpin works only on selected (yellow) point
                    target_idx = tarp.get_tieout_indices()[selected_tieout_index]
                    if target_idx in tarp.pinned_points: 
                        del tarp.pinned_points[target_idx]
                
                elif event.key == pygame.K_LEFT: selected_tieout_index = (selected_tieout_index - 1) % 9
                elif event.key == pygame.K_RIGHT: selected_tieout_index = (selected_tieout_index + 1) % 9
                elif event.key == pygame.K_SPACE: paused = not paused
        
        drag_info_for_update = None
        if dragged_tieout_info:
            mx, my = pygame.mouse.get_pos()
            world_pos = gluUnProject(mx, WINDOW_HEIGHT-my, dragged_tieout_info['depth'], modelview_matrix, projection_matrix, viewport)
            if world_pos:
                drag_info_for_update = {'index': dragged_tieout_info['index'], 'target_pos': np.array(world_pos)}
        
        if not paused:
            tarp.update(dt, drag_info_for_update)
            
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        if is_ortho_view:
            glMatrixMode(GL_PROJECTION); glLoadIdentity()
            aspect = WINDOW_WIDTH / WINDOW_HEIGHT
            glOrtho(-ortho_zoom*aspect, ortho_zoom*aspect, -ortho_zoom, ortho_zoom, -100, 100)
            glMatrixMode(GL_MODELVIEW); glLoadIdentity()
            gluLookAt(ortho_pan[0], 10, ortho_pan[1], ortho_pan[0], 0, ortho_pan[1], 0, 0, -1)
        else:
            glMatrixMode(GL_PROJECTION); glLoadIdentity()
            gluPerspective(45, (WINDOW_WIDTH/WINDOW_HEIGHT), 0.1, 50.0)
            glMatrixMode(GL_MODELVIEW); glLoadIdentity()
            glTranslatef(0.0, -1.0, -persp_dist)
            glRotatef(persp_rot[1], 1, 0, 0); glRotatef(persp_rot[0], 0, 1, 0)

        draw_ground()
        drag_idx = dragged_tieout_info['index'] if dragged_tieout_info else None
        tarp.draw(selected_tieout_index, drag_idx)

        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        gluOrtho2D(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        controls_text = [
            f"View: {'Top-Down' if is_ortho_view else '3D Orbit'} (V to switch)",
            "L-Click+Drag: Pan/Orbit", "R-Click Tie-out: Toggle Drag",
            "S: Pin in place", "U: Un-pin selected", "R: Reset", "SPACE: Pause", "",
            "Press S while dragging to pin a point anywhere!", "",
            f"Selected Tie-out: {selected_tieout_index+1}/9",
            "Status: " + ("DRAGGING" if dragged_tieout_info else "Paused" if paused else "Running")]
        for i, line in enumerate(controls_text):
            draw_text(10, WINDOW_HEIGHT - 25 * (i + 1), line, font)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW); glPopMatrix()
        
        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    main()