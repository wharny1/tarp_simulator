import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

# --- Configuration Constants ---
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
TARP_SIZE = 3.0  # Represents a 3x3 meter tarp
POINTS_PER_AXIS = 17 # Higher resolution for smoother cloth
SOLVER_ITERATIONS = 5 # More iterations = stiffer cloth
GRAVITY = np.array([0.0, -9.8, 0.0])
POLE_HEIGHT = 1.2 # Height for lifting tie-outs

# --- Tarp Class ---
class Tarp:
    def __init__(self, size, num_points):
        self.size = size
        self.num_points = num_points
        self.points = np.zeros((num_points, num_points, 3), dtype=float)
        self.old_points = np.zeros_like(self.points)
        self.pinned_points = {} # { (i, j): position_vector }

        # Generate a list of spring constraints: (point1_idx, point2_idx, rest_length)
        self.springs = []
        step = size / (num_points - 1)
        for i in range(num_points):
            for j in range(num_points):
                # Structural springs
                if i < num_points - 1:
                    self.springs.append(((i, j), (i + 1, j), step))
                if j < num_points - 1:
                    self.springs.append(((i, j), (i, j + 1), step))
                # Shear springs (prevent squashing)
                if i < num_points - 1 and j < num_points - 1:
                    self.springs.append(((i, j), (i + 1, j + 1), np.sqrt(2) * step))
                    self.springs.append(((i + 1, j), (i, j + 1), np.sqrt(2) * step))
        
        self.reset()
    
    def reset(self):
        """ 
        Resets the tarp to a FACE-ON view.
        It's created on the X-Y plane (like the screen) and pinned at the top.
        """
        self.pinned_points.clear()
        step = self.size / (self.num_points - 1)
        
        # CHANGE 1: Create the tarp vertically on the X-Y plane, centered at (0,0,0)
        for i in range(self.num_points):
            for j in range(self.num_points):
                # Y coordinate goes from +size/2 down to -size/2
                # X coordinate goes from -size/2 across to +size/2
                # Z coordinate is 0, so it's flat to the camera
                y_pos = (self.size / 2) - (i * step)
                x_pos = (-self.size / 2) + (j * step)
                self.points[i, j] = np.array([x_pos, y_pos, 0.0])
        self.old_points = np.copy(self.points)

        # CHANGE 2: Pin the top two corners so it hangs in place and is visible.
        end = self.num_points - 1
        top_left_idx = (0, 0)
        top_right_idx = (0, end)
        self.pinned_points[top_left_idx] = np.copy(self.points[top_left_idx])
        self.pinned_points[top_right_idx] = np.copy(self.points[top_right_idx])


    def get_tieout_indices(self):
        """ Returns the grid indices for the 9 main tie-outs. """
        mid = (self.num_points - 1) // 2
        end = self.num_points - 1
        return [
            (0, 0), (0, mid), (0, end),
            (mid, 0), (mid, mid), (mid, end),
            (end, 0), (end, mid), (end, end)
        ]

    def update(self, dt):
        """ Physics update using Verlet Integration. """
        for i in range(self.num_points):
            for j in range(self.num_points):
                if (i, j) not in self.pinned_points:
                    temp_pos = np.copy(self.points[i, j])
                    velocity = self.points[i, j] - self.old_points[i, j]
                    self.points[i, j] += velocity + GRAVITY * dt * dt
                    self.old_points[i, j] = temp_pos
        
        for _ in range(SOLVER_ITERATIONS):
            for p1_idx, p2_idx, rest_length in self.springs:
                p1 = self.points[p1_idx]
                p2 = self.points[p2_idx]
                delta = p2 - p1
                dist = np.linalg.norm(delta)
                if dist > 0:
                    diff = (dist - rest_length) / dist
                    correction = delta * 0.5 * diff
                    
                    if p1_idx not in self.pinned_points:
                        self.points[p1_idx] += correction
                    if p2_idx not in self.pinned_points:
                        self.points[p2_idx] -= correction
            
            for idx, pos in self.pinned_points.items():
                self.points[idx] = pos
            
            # Ground constraint (now less visible, but still works)
            self.points[:,:,1] = np.maximum(self.points[:,:,1], -self.size*2) # Lower ground

    def draw(self, selected_tieout_idx):
        """ Draws the tarp mesh, tie-outs, and guylines. """
        glColor3f(0.0, 0.4, 0.2)
        glBegin(GL_QUADS)
        for i in range(self.num_points - 1):
            for j in range(self.num_points - 1):
                glVertex3fv(self.points[i, j])
                glVertex3fv(self.points[i + 1, j])
                glVertex3fv(self.points[i + 1, j + 1])
                glVertex3fv(self.points[i, j + 1])
        glEnd()
        
        glColor3f(1.0, 1.0, 1.0)
        for i in range(self.num_points - 1):
            for j in range(self.num_points - 1):
                glBegin(GL_LINE_LOOP)
                glVertex3fv(self.points[i, j])
                glVertex3fv(self.points[i + 1, j])
                glVertex3fv(self.points[i + 1, j + 1])
                glVertex3fv(self.points[i, j + 1])
                glEnd()

        all_tieouts = self.get_tieout_indices()
        for i, tieout_idx in enumerate(all_tieouts):
            pos = self.points[tieout_idx]
            if i == selected_tieout_idx:
                glColor3f(1.0, 1.0, 0.0); glPointSize(15)
            else:
                glColor3f(1.0, 0.5, 0.0); glPointSize(10)

            glBegin(GL_POINTS)
            glVertex3fv(pos)
            glEnd()

            if tieout_idx in self.pinned_points:
                glColor3f(0.8, 0.8, 0.8)
                glBegin(GL_LINES)
                glVertex3fv(pos)
                glVertex3fv(self.pinned_points[tieout_idx])
                glEnd()

# Helper Functions (no changes needed)
def draw_ground():
    glColor3f(0.3, 0.3, 0.3)
    glBegin(GL_LINES)
    for i in np.arange(-5, 5.1, 0.5):
        glVertex3f(i, -3, -5); glVertex3f(i, -3, 5) # Lowered ground plane
        glVertex3f(-5, -3, i); glVertex3f(5, -3, i)
    glEnd()

def draw_text(x, y, text, font):
    text_surface = font.render(text, True, (255, 255, 255), (0, 0, 0, 128))
    text_data = pygame.image.tostring(text_surface, "RGBA", True)
    glWindowPos2d(x, y)
    glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

# --- Main Application ---
def main():
    pygame.init()
    pygame.display.set_caption("3x3 Tarp Simulator")
    display = (WINDOW_WIDTH, WINDOW_HEIGHT)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    font = pygame.font.Font(None, 24)

    glEnable(GL_DEPTH_TEST)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    
    # Camera state
    camera_distance = 5.0 
    # CHANGE 3: Set camera to have ZERO rotation for a perfect face-on view.
    camera_rotation = [0, 0] 
    
    tarp = Tarp(TARP_SIZE, POINTS_PER_AXIS)
    clock = pygame.time.Clock()
    dt = 1/60.0

    selected_tieout_index = 4
    paused = False
    mouse_dragging = False
    last_mouse_pos = (0, 0)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: mouse_dragging = True; last_mouse_pos = event.pos
                elif event.button == 4: camera_distance = max(2.0, camera_distance - 0.5)
                elif event.button == 5: camera_distance = min(20.0, camera_distance + 0.5)
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: mouse_dragging = False
            if event.type == pygame.MOUSEMOTION and mouse_dragging:
                dx, dy = event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1]
                camera_rotation[0] += dx * 0.2
                camera_rotation[1] += dy * 0.2
                camera_rotation[1] = max(-89.9, min(89.9, camera_rotation[1]))
                last_mouse_pos = event.pos
            if event.type == pygame.KEYDOWN:
                current_tieout = tarp.get_tieout_indices()[selected_tieout_index]
                if event.key == pygame.K_LEFT: selected_tieout_index = (selected_tieout_index - 1) % 9
                if event.key == pygame.K_RIGHT: selected_tieout_index = (selected_tieout_index + 1) % 9
                if event.key == pygame.K_s:
                    if current_tieout not in tarp.pinned_points:
                        pos = np.copy(tarp.points[current_tieout]); pos[1] = 0.0
                        tarp.pinned_points[current_tieout] = pos
                if event.key == pygame.K_l:
                     if current_tieout not in tarp.pinned_points:
                        pos = np.copy(tarp.points[current_tieout]); pos[1] = POLE_HEIGHT
                        tarp.pinned_points[current_tieout] = pos
                if event.key == pygame.K_u:
                    if current_tieout in tarp.pinned_points: del tarp.pinned_points[current_tieout]
                if event.key == pygame.K_r: tarp.reset()
                if event.key == pygame.K_SPACE: paused = not paused
        
        if not paused:
            tarp.update(dt)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # CHANGE 4: Set the camera to pivot around the origin (0,0,0) where the tarp is centered.
        glTranslatef(0.0, 0.0, -camera_distance)
        glRotatef(camera_rotation[1], 1, 0, 0) # Tilt
        glRotatef(camera_rotation[0], 0, 1, 0) # Pan

        # Draw scene (ground is now below the hanging tarp)
        draw_ground()
        tarp.draw(selected_tieout_index)

        # Draw UI Text
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        gluOrtho2D(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        controls_text = [
            "[3x3 Tarp Simulator]", "Left Click + Drag: Orbit Camera", "Mouse Wheel: Zoom",
            "Left/Right Arrows: Select Tie-out", "S: Stake selected tie-out to ground",
            "L: Lift selected tie-out (pole)", "U: Un-pin selected tie-out", "R: Reset Tarp",
            "SPACE: Pause Simulation", "", f"Selected Tie-out: {selected_tieout_index+1}/9",
            "Paused" if paused else "Running"]
        for i, line in enumerate(controls_text):
            draw_text(10, WINDOW_HEIGHT - 25 * (i + 1), line, font)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW); glPopMatrix()
        
        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    main()