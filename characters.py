import random
import pygame
import math
from util import *


class Player:

    def __init__(self, world_width, world_height, walls) -> None:
        # Player settings
        self.size = 50
        self.speed = 5

        # Player initial position in the world (center of the larger world)
        self.rect = None

        self.x = world_width // 2
        self.y = world_height // 2
        
        while True:
            self.rect = pygame.Rect(self.x, self.y, self.size, self.size)
            
            if check_collision(self.rect, walls):
                self.x += random.randint(-5, 5)
                self.y += random.randint(-5, 5)
            else:
                break

        self.score = 0
        self.ammo = 10
        self.health = 5

        self.images = {}

        for direction in ('up', 'down', 'left', 'right'):
            image = pygame.image.load(f'images/player_{direction}.png')
            self.images[direction] = pygame.transform.scale(image, (self.size, self.size))

        self.direction = "up"

    def draw(self, screen, camera_x, camera_y):
        # player_image = self.player.images[self.player.direction]
        # self.player.rect = player_image.get_rect(center=(self.player.x, self.player.y))
        screen.blit(self.images[self.direction], (self.x - camera_x, self.y - camera_y))


class Zombie:
    def __init__(self, world_width, world_height, size=50, speed=1):
        # Spawn the zombie at a random position around the edges of the map
        self.size = size
        self.zombie_color = (0, 255, 0)
        self.world_width = world_width
        self.world_height = world_height
        self.speed = speed

        self.x, self.y = self.spawn()

        self.images = {}

        for direction in ('up', 'down', 'left', 'right'):
            image = pygame.image.load(f'images/zombie_{direction}.png')
            self.images[direction] = pygame.transform.scale(image, (self.size, self.size))

        self.direction = "up"

        self.rect = pygame.Rect(0, 0, self.size, self.size)
        self.rect.center = (self.x, self.y)

    def spawn(self):
        """Spawns a zombie at a random location around the edges of the world."""
        spawn_positions = [
            (random.randint(0, self.world_width - self.size), 0),  # Top edge
            (random.randint(0, self.world_width - self.size), self.world_height - self.size),  # Bottom edge
            (0, random.randint(0, self.world_height - self.size)),  # Left edge
            (self.world_width - self.size, random.randint(0, self.world_height - self.size))  # Right edge
        ]
        return random.choice(spawn_positions)


    def move_toward_player(self, player_x, player_y, walls):
        """Moves the zombie toward the player's position, with better handling of wall collisions."""
        dx, dy = player_x - self.x, player_y - self.y
        distance = math.hypot(dx, dy)
        if distance != 0:
            dx, dy = dx / distance, dy / distance  # Normalize
        
        # Try horizontal movement first
        new_x = self.x + dx * self.speed
        new_rect = pygame.Rect(new_x, self.y, self.size, self.size)
        can_move_x = not check_collision(new_rect, walls)

        if can_move_x:
            self.x = new_x
        else:
            # Increase speed along y-axis if horizontal movement is blocked
            new_y = self.y + dy * self.speed * 1.5  # Increase speed in vertical direction
            new_rect = pygame.Rect(self.x, new_y, self.size, self.size)
            if not check_collision(new_rect, walls):
                self.y = new_y

        # Try vertical movement next
        new_y = self.y + dy * self.speed
        new_rect = pygame.Rect(self.x, new_y, self.size, self.size)
        can_move_y = not check_collision(new_rect, walls)

        if can_move_y:
            self.y = new_y
        else:
            # Increase speed along x-axis if vertical movement is blocked
            new_x = self.x + dx * self.speed * 1.5  # Increase speed in horizontal direction
            new_rect = pygame.Rect(new_x, self.y, self.size, self.size)
            if not check_collision(new_rect, walls):
                self.x = new_x

        # Update the zombie's position and direction
        self.rect.topleft = (self.x, self.y)

        # Set direction for zombie based on movement
        if abs(dx) > abs(dy):
            if dx > 0:
                self.direction = 'right'
            else:
                self.direction = 'left'
        else:
            if dy > 0:
                self.direction = 'down'
            else:
                self.direction = 'up'


    def draw(self, screen, camera_x, camera_y):
        """Draws the zombie as a green rectangle."""
        # pygame.draw.rect(screen, self.zombie_color, self.rect)
        # pygame.draw.rect(screen, self.zombie_color, (self.rect.x - camera_x, self.rect.y - camera_y, self.size, self.size))

        # zombie.rect = zombie_image.get_rect(center=(zombie.x, zombie.y))
        screen.blit(self.images[self.direction], (self.rect.x - camera_x, self.rect.y - camera_y))

