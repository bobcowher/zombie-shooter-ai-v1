# Bullet class

import pygame

class Bullet:
    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.direction = direction

        print(f"Bullet launched in direction {direction}")
        self.rect = pygame.Rect(x, y, 10, 10)  # Size of the bullet
        self.bullet_speed = 10
        self.bullet_color = (192, 192, 192)

    def move(self):
        if self.direction == "up":
            self.y -= self.bullet_speed
        elif self.direction == "down":
            self.y += self.bullet_speed
        elif self.direction == "left":
            self.x -= self.bullet_speed
        elif self.direction == "right":
            self.x += self.bullet_speed
        self.rect.topleft = (self.x, self.y)

    def draw(self, screen, camera_x, camera_y):
        pygame.draw.rect(screen, self.bullet_color, (self.rect.x - camera_x, self.rect.y - camera_y, 10, 10))