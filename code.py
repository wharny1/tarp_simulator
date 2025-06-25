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
        """ Resets the tarp to a flat state, high in the air. """
        self.pinned_points.clear()
        step = self.size / (self.num_points - 1)
        for i in range(self.num_points):
            for j in range(self.num_points):
                self.points[i, j] = np.array([-self.size/2 + j*step, 3.0, -self.size/2 + i*step])
        self.old_points = np.copy(self.points)

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
        # 1. Verlet integration step
        for i in range(self.num_points):
            for j in range(self.num_points):
                if (i, j) not in self.pinned_points:
                    temp_pos = np.copy(self.points[i, j])
                    velocity = self.points[i, j] - self.old_points[i, j]
                    self.points[i, j] += velocity + GRAVITY * dt * dt
                    self.old_points[i, j] = temp_pos
        
        # 2. Constraint satisfaction (iterated for stability)
        for _ in range(SOLVER_ITERATIONS):
            # Spring constraints
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
            
            # Pinned point constraints
            for idx, pos in self.pinned_points.items():
                self.points[idx] = pos
            
            # Ground constraint
            self.points[:,:,1] = np.maximum(self.points[:,:,1], 0.0)

    def draw(self, selected_tieout_idx):
        """ Draws the tarp mesh, tie-outs, and guylines. """
        # Draw Tarp Surface (with lighting)
        glEnable(GL_LIGHTING)
        glColor3f(0.0, 0.4, 0.2) # Tarp color (dark green)
        glBegin(GL_QUADS)
        for i in range(self.num_points - 1):
            for j in range(self.num_points - 1):
                p1 = self.points[i, j]
                p2 = self.points[i + 1, j]
                p3 = self.points[i + 1, j + 1]
                p4 = self.points[i, j + 1]
                
                # Calculate normal for lighting
                v1 = p2 - p1
                v2 = p4 - p1
                normal = np.cross(v1, v2)
                normal /= np.linalg.norm(normal)
                glNormal3fv(normal)
                
                glVertex3fv(p1)
                glVertex3fv(p2)
                glVertex3fv(p3)
                glVertex3fv(p4)
        glEnd()
        glDisable(GL_LIGHTING)

        # Draw Tie-outs and Guylines
        all_tieouts = self.get_tieout_indices()
        for i, tieout_idx in enumerate(all_tieouts):
            pos = self.points[tieout_idx]
            
            # Highlight selected tie-out
            if i == selected_tieout_idx:
                glColor3f(1.0, 1.0, 0.0) # Yellow for selected
                glPointSize(15)
            else:
                glColor3f(1.0, 0.5, 0.0) # Orange for others
                glPointSize(10)

            glBegin(GL_POINTS)
            glVertex3fv(pos)
            glEnd()

            # Draw "guyline" if pinned
            if tieout_idx in self.pinned_points:
                glColor3f(0.8, 0.8, 0.8) # Grey for guylines
                glBegin(GL_LINES)
                glVertex3fv(pos)
                glVertex3fv(self.pinned_points[tieout_idx])
                glEnd()

# --- Helper Functions ---
def draw_ground():
    """ Draws a grid on the X-Z plane to represent the ground. """
    glColor3f(0.3, 0.3, 0.3)
    glBegin(GL_LINES)
    for i in np.arange(-5, 5.1, 0.5):
        glVertex3f(i, 0, -5)
        glVertex3f(i, 0, 5)
        glVertex3f(-5, 0, i)
        glVertex3f(5, 0, i)
    glEnd()

def draw_text(x, y, text, font):
    """ Renders text in a 2D overlay. """
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
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (5, 5, 5, 1))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.7, 0.7, 0.7, 1))
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    
    # Camera state
    camera_distance = 6.0
    camera_rotation = [45, -30] # [pan, tilt]
    
    tarp = Tarp(TARP_SIZE, POINTS_PER_AXIS)
    clock = pygame.time.Clock()
    dt = 1/60.0

    selected_tieout_index = 4 # Start with the center tie-out
    paused = False

    mouse_dragging = False
    last_mouse_pos = (0, 0)
    
    # Main loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            # --- Mouse Controls ---
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    mouse_dragging = True
                    last_mouse_pos = event.pos
                elif event.button == 4: # Scroll up
                    camera_distance = max(2.0, camera_distance - 0.5)
                elif event.button == 5: # Scroll down
                    camera_distance = min(20.0, camera_distance + 0.5)
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_dragging = False
            if event.type == pygame.MOUSEMOTION and mouse_dragging:
                dx, dy = event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1]
                camera_rotation[0] += dx * 0.2
                camera_rotation[1] += dy * 0.2
                camera_rotation[1] = max(-89, min(89, camera_rotation[1])) # Clamp tilt
                last_mouse_pos = event.pos
            # --- Keyboard Controls ---
            if event.type == pygame.KEYDOWN:
                current_tieout = tarp.get_tieout_indices()[selected_tieout_index]
                
                if event.key == pygame.K_LEFT:
                    selected_tieout_index = (selected_tieout_index - 1) % 9
                if event.key == pygame.K_RIGHT:
                    selected_tieout_index = (selected_tieout_index + 1) % 9

                if event.key == pygame.K_s: # Stake to ground
                    if current_tieout not in tarp.pinned_points:
                        pos = np.copy(tarp.points[current_tieout])
                        pos[1] = 0.0 # Pin to ground level
                        tarp.pinned_points[current_tieout] = pos
                
                if event.key == pygame.K_l: # Lift with pole
                     if current_tieout not in tarp.pinned_points:
                        pos = np.copy(tarp.points[current_tieout])
                        pos[1] = POLE_HEIGHT # Pin to pole height
                        tarp.pinned_points[current_tieout] = pos

                if event.key == pygame.K_u: # Unpin
                    if current_tieout in tarp.pinned_points:
                        del tarp.pinned_points[current_tieout]
                        
                if event.key == pygame.K_r: # Reset
                    tarp.reset()
                if event.key == pygame.K_SPACE: # Pause
                    paused = not paused
        
        # Physics Update
        if not paused:
            tarp.update(dt)

        # Rendering
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Apply camera transformations
        glTranslatef(0.0, -1.0, -camera_distance)
        glRotatef(camera_rotation[1], 1, 0, 0)
        glRotatef(camera_rotation[0], 0, 1, 0)

        # Draw scene
        draw_ground()
        tarp.draw(selected_tieout_index)

        # Draw UI Text
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)

        controls_text = [
            "[3x3 Tarp Simulator]",
            "Left Click + Drag: Orbit Camera",
            "Mouse Wheel: Zoom",
            "Left/Right Arrows: Select Tie-out",
            "S: Stake selected tie-out to ground",
            "L: Lift selected tie-out (pole)",
            "U: Un-pin selected tie-out",
            "R: Reset Tarp",
            "SPACE: Pause Simulation",
            "",
            f"Selected Tie-out: {selected_tieout_index+1}/9",
            "Paused" if paused else "Running"
        ]
        for i, line in enumerate(controls_text):
            draw_text(10, WINDOW_HEIGHT - 25 * (i + 1), line, font)
            
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    main()