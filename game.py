import pygame
import sys
from bullet import Bullet
from characters import Player, Zombie
import random
from util import check_collision, get_collision
from walls import *

class ZombieShooter:

    def __init__(self, window_width, window_height, world_height, world_width, fps, sound=False):

        self.window_width = window_width
        self.window_height = window_height
        self.world_height = world_height
        self.world_width = world_width

        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height))

        pygame.display.set_caption('Zombie Shooter')

        self.font = pygame.font.SysFont(None, 36)  # Font size 36

        self.clock = pygame.time.Clock() 
        self.fps = fps

        self.walls = walls_1

        self.player = Player(world_height=self.world_height, world_width=self.world_width, walls=self.walls)

        self.background_color = (181, 101, 29) # Light brown
        self.wall_color = (1, 50, 32)
        self.border_color = (255, 0, 0)

        self.announcement_font = pygame.font.SysFont(None, 100)

        self.bullets = []
        self.zombies = []

        self.zombie_top_speed = 1

        self.level_goal = 5

        self.max_zombie_count = 5

        self.level = 1

        self.sound = sound

        if self.sound:
            pygame.mixer.pre_init(44100, -16, 2, 64)
            pygame.mixer.init()
            pygame.mixer.music.load("sounds/background_music.wav")
            pygame.mixer.music.play(-1,0.0)

            self.last_walk_play_time = 0

            self.zombie_bite = pygame.mixer.Sound("sounds/zombie_bite_1.wav")
            self.zombie_hit = pygame.mixer.Sound("sounds/zombie_hit.wav")
            self.shotgun_blast = pygame.mixer.Sound("sounds/shotgun_blast.wav")
            self.zombie_snarl = pygame.mixer.Sound("sounds/zombie_snarl.wav")
            self.footstep = pygame.mixer.Sound("sounds/footstep.wav")
            self.vocals_1 = pygame.mixer.Sound("sounds/one_of_those_things_got_in.wav")
            self.vocals_2 = pygame.mixer.Sound("sounds/virus_infection_alert.wav")
            self.vocals_3 = pygame.mixer.Sound("sounds/come_and_see.wav")
            self.vocals_4 = pygame.mixer.Sound("sounds/no_escape.wav")

            self.vocals_1.play()

    def play_walking_sound(self):
        if self.sound:
            current_time = pygame.time.get_ticks()
            if(current_time - self.last_walk_play_time > 1000):
                self.footstep.play()
                self.last_walk_play_time = current_time


    def start_next_level(self):
        self.level += 1
        
        if self.level > 3:
            next_level_surface = self.announcement_font.render('You Won!', True, (255, 0, 0))
        else:
            next_level_surface = self.announcement_font.render(f'Starting level {self.level}', True, (255, 0, 0))

        next_level_rect = next_level_surface.get_rect(center=(self.window_width // 2, self.window_height // 2))

        self.zombies = []
        self.bullets = []

        if self.level == 2:
            self.vocals_2.play()
            self.walls = walls_2
            self.level_goal = 15
        elif self.level == 3:
            self.vocals_3.play()
            self.walls = walls_3
            self.level_goal = 30

        self.screen.blit(next_level_surface, next_level_rect)
        
        self.zombie_top_speed += 1
        self.max_zombie_count += 2
        # self.player.score = 0

        self.player = Player(world_height=self.world_height, world_width=self.world_width, walls=self.walls)

        pygame.display.flip()

        pygame.time.wait(4000)

        if self.level > 3:
            # Quit the game
            pygame.quit()
            sys.exit()



        


    def game_over(self):
        # Render the "You Died" message
        game_over_surface = self.announcement_font.render('You Died', True, (255, 0, 0))  # Red text
        game_over_rect = game_over_surface.get_rect(center=(self.window_width // 2, self.window_height // 2))

        # Blit the message to the screen
        self.screen.blit(game_over_surface, game_over_rect)

        # Update the display to show the message
        pygame.display.flip()

        self.zombie_snarl.play()

        # Pause for 2 seconds (2000 milliseconds) before quitting
        pygame.time.wait(2000)

        # Quit the game
        pygame.quit()
        sys.exit()

    def fill_background(self):
        self.screen.fill(self.background_color)  # Fill the screen with white (background)

        score_surface = self.font.render(f'Score: {self.player.score}', True, (0, 0, 0))  # Render the score with black color
        self.screen.blit(score_surface, (10, 10))  # Draw the score at the top-left corner (10, 10)
        health_surface = self.font.render(f'Health: {self.player.health}', True, (0, 0, 0))  # Render the score with black color
        self.screen.blit(health_surface, (10, 35))  # Draw the score at the top-left corner (10, 10)
        level_surface = self.font.render(f'Level: {self.level}', True, (0, 0, 0))  # Render the score with black color
        self.screen.blit(level_surface, (10, 60))  # Draw the score at the top-left corner (10, 10)

    def fire_bullet(self):
        bullet = Bullet(self.player.x, self.player.y, self.player.direction)
        self.bullets.append(bullet)
        self.shotgun_blast.play()

        print("Space pressed. Bullet fired")

    def step(self):
            
            player_moved = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                # Shooting event: spacebar to fire self.bullets
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # bullet = Bullet(player_x + player_size // 2, player_y + player_size // 2, player.direction)
                        self.fire_bullet()

            if len(self.zombies) < self.max_zombie_count and random.randint(1, 100) < 3:  # 3% chance of spawning a zombie per frame
                self.zombies.append(Zombie(world_height=self.world_height, world_width=self.world_width, size=80, speed=random.randint(1,self.zombie_top_speed)))  # Instantiate a new zombie

            # Get key presses
            keys = pygame.key.get_pressed()
                
            new_player_x = self.player.x
            if keys[pygame.K_a]:  # Left
                new_player_x -= self.player.speed
                self.player.direction = "left"
            if keys[pygame.K_d]:  # Right
                new_player_x += self.player.speed
                self.player.direction = "right"

            new_player_rect = pygame.Rect(new_player_x, self.player.y, self.player.size, self.player.size)

            collision = check_collision(new_player_rect, self.walls)

            if not collision and self.player.x != new_player_x:
                self.player.x = new_player_x
                self.play_walking_sound()
            

            new_player_y = self.player.y
            if keys[pygame.K_w]:  # Up
                new_player_y -= self.player.speed
                self.player.direction = "up"
            if keys[pygame.K_s]:  # Down
                new_player_y += self.player.speed
                self.player.direction = "down"

            new_player_rect = pygame.Rect(self.player.x, new_player_y, self.player.size, self.player.size)

            collision = check_collision(new_player_rect, self.walls)

            if not collision and self.player.y != new_player_y:
                self.player.y = new_player_y
                self.play_walking_sound()
                
            self.player.rect = pygame.Rect(self.player.x, self.player.y, self.player.size, self.player.size)
            # Check for collision with walls
            collision = False
            
            # Update camera position (centered on player)
            camera_x = self.player.x - self.window_width // 2
            camera_y = self.player.y - self.window_height // 2

            # Keep camera within world bounds
            camera_x = max(0, min(camera_x, self.world_width - self.window_width))
            camera_y = max(0, min(camera_y, self.world_height - self.window_height))


            # Move self.zombies toward player and check for collisions with self.bullets
            self.zombies_temp = []
            for zombie in self.zombies:
                if check_collision(zombie.rect, self.bullets):
                    bullet = get_collision(zombie.rect, self.bullets)
                    self.player.score += 1
                    self.bullets.remove(bullet)
                    if self.sound:
                        self.zombie_hit.play()
                    bullet = None
                elif check_collision(zombie.rect, [self.player.rect]):
                    self.player.health -= 1
                    
                    if self.sound:
                        self.zombie_bite.play()

                else:
                    self.zombies_temp.append(zombie)
            
            self.zombies = self.zombies_temp


            for zombie in self.zombies:
                zombie.move_toward_player(self.player.x, self.player.y, self.walls)

            # Drawing
            self.fill_background()


            # Move and draw self.bullets
            for bullet in self.bullets:
                bullet.move()
                bullet.draw(self.screen, camera_x, camera_y)
                
                if check_collision(bullet.rect, self.walls):
                    self.bullets.remove(bullet)

            # Draw self.zombies
            # for zombie in self.zombies:
            #     zombie.draw(screen, camera_x, camera_y)

            # Draw the player (adjusted for the camera position)
            self.player.draw(self.screen, camera_x, camera_y)

            for zombie in self.zombies:
                zombie.draw(self.screen, camera_x, camera_y)
            
            # Draw the world boundaries for testing
            pygame.draw.rect(self.screen, self.border_color, (0 - camera_x, 0 - camera_y, self.world_width, self.world_height), 5)

            for wall in self.walls:
                pygame.draw.rect(self.screen, self.wall_color, (wall.x - camera_x, wall.y - camera_y, wall.width, wall.height))

            # Update the display
            pygame.display.flip()

            if self.player.health <= 0:
                self.game_over()

            # Cap the frame rate
            self.clock.tick(self.fps)

            if(self.level_goal <= self.player.score):
                self.start_next_level()