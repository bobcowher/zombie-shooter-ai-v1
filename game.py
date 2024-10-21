import pygame
import sys
from bullet import SingleBullet, ShotgunBullet
from assets import Player, Zombie, TreasureChest, HealthDrop
import random
from util import check_collision, get_collision
from walls import *
import cv2
import numpy as np


class ZombieShooter:

    def __init__(self, window_width, window_height, world_height, world_width, fps, sound=False, render_mode="human"):

        self.window_width = window_width
        self.window_height = window_height
        self.world_height = world_height
        self.world_width = world_width

        if render_mode == "human":
            self.human = True
            self.sound = sound
        else:
            self.human = False
            self.sound = False

        self.treasure_chest = None  # No chest initially
        self.health_drop = None  # No health drop initially

        self.paused = False  # Game starts unpaused

        self.gun_type = "shotgun"  # Start with shotgun
        self.fire_mode = "single"  # Add this to __init__
        self.shotgun_ammo = 5  # Start with 10 shotgun shells
        self.out_of_ammo_message_displayed = False  # Initialize in __init__()

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

        self.done = False



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

    def toggle_pause(self):
        self.paused = not self.paused  # Toggle between paused and unpaused

        if self.paused:
            pause_surface = self.announcement_font.render('Game Paused', True, (255, 255, 255))  # White text
            pause_rect = pause_surface.get_rect(center=(self.window_width // 2, self.window_height // 2))
            self.screen.blit(pause_surface, pause_rect)
            pygame.display.flip()  # Update display to show pause message

            # Wait until the player unpauses (ignore everything else)
            while self.paused:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self.paused = False  # Unpause the game
                self.clock.tick(10)  # Prevent busy-waiting

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
            if self.sound:
                self.vocals_2.play()
            self.walls = walls_2
            self.level_goal = 15
        elif self.level == 3:
            if self.sound:
                self.vocals_3.play()
            self.walls = walls_3
            self.level_goal = 30

        self.screen.blit(next_level_surface, next_level_rect)

        # Spawn treasure chest at a random location
        x, y = random.randint(50, self.world_width - 50), random.randint(50, self.world_height - 50)
        self.treasure_chest = TreasureChest(x, y)

        self.zombie_top_speed += 1
        self.max_zombie_count += 2

        self.player = Player(world_height=self.world_height, world_width=self.world_width, walls=self.walls)

        pygame.display.flip()
        pygame.time.wait(4000)

        if self.level > 3:
            self.done = True

            # if self.human:
            #     pygame.quit()
            #     sys.exit()

    def game_over(self):
        # Render the "You Died" message
        game_over_surface = self.announcement_font.render('You Died', True, (255, 0, 0))  # Red text
        game_over_rect = game_over_surface.get_rect(center=(self.window_width // 2, self.window_height // 2))

        # Blit the message to the screen
        self.screen.blit(game_over_surface, game_over_rect)

        # Update the display to show the message
        pygame.display.flip()

        self.done = True

        if self.sound:
            self.zombie_snarl.play()

        # Pause for 2 seconds (2000 milliseconds) before quitting
        if self.human:
            pygame.time.wait(2000)

            # # Quit the game
            # pygame.quit()
            # sys.exit()

    def fill_background(self):
        self.screen.fill(self.background_color)

        # Display score, health, level, ammo, and gun type
        score_surface = self.font.render(f'Score: {self.player.score}', True, (0, 0, 0))
        self.screen.blit(score_surface, (10, 10))

        health_surface = self.font.render(f'Health: {self.player.health}', True, (0, 0, 0))
        self.screen.blit(health_surface, (10, 35))

        level_surface = self.font.render(f'Level: {self.level}', True, (0, 0, 0))
        self.screen.blit(level_surface, (10, 60))

        ammo_surface = self.font.render(f'Shotgun Ammo: {self.shotgun_ammo}', True, (0, 0, 0))
        self.screen.blit(ammo_surface, (10, 85))

        gun_type_surface = self.font.render(f'Gun: {self.gun_type.title()}', True, (0, 0, 0))
        self.screen.blit(gun_type_surface, (10, 110))

        # Display out of ammo message if needed
        if self.out_of_ammo_message_displayed and self.gun_type == "shotgun":
            out_of_ammo_surface = self.font.render(
                "Out of shotgun ammo! Press TAB to switch to single shot.", True, (255, 0, 0)
            )
            out_of_ammo_rect = out_of_ammo_surface.get_rect(center=(self.window_width // 2, self.window_height // 2))
            self.screen.blit(out_of_ammo_surface, out_of_ammo_rect)

    def fire_single_bullet(self):
        bullet = SingleBullet(self.player.x, self.player.y, self.player.direction)
        self.bullets.append(bullet)

        if self.sound:
            self.shotgun_blast.play()

        print("Space pressed. Bullet fired")


    def fire_shotgun_bullet(self):
        if self.shotgun_ammo > 0:
            directions = [
                (self.player.direction, 0),    # Main bullet (straight)
                (self.player.direction, -10),  # Left bullet (angled)
                (self.player.direction, 10)    # Right bullet (angled)
            ]

            for direction, angle_offset in directions:
                bullet = ShotgunBullet(self.player.x, self.player.y, direction, angle_offset)
                self.bullets.append(bullet)

            self.shotgun_ammo -= 1  # Decrease ammo
            if self.sound:
                self.shotgun_blast.play()
            print(f"Shotgun fired. Remaining ammo: {self.shotgun_ammo}")
            self.out_of_ammo_message_displayed = False  # Hide the message
        else:
            print("Out of shotgun ammo!")
            self.out_of_ammo_message_displayed = True  # Show the message

    def _get_info(self):

        
        gun_type_num = 1 if "single" in self.gun_type else 2

        return {
            "health" : self.player.health,
            "shotgun_ammo" : self.shotgun_ammo,
            "gun_type" : self.gun_type,
            "gun_type_num" : gun_type_num
        }


    def _get_obs(self):
        # Capture the surface to an RGB array
        screen_array = pygame.surfarray.array3d(self.screen)

        # Transpose array from (width, height, channels) to (height, width, channels)
        screen_array = np.transpose(screen_array, (1, 0, 2))

        # Convert to BGR for OpenCV compatibility
        screen_array_bgr = cv2.cvtColor(screen_array, cv2.COLOR_RGB2BGR)

        # Convert to grayscale using OpenCV
        grayscale = cv2.cvtColor(screen_array_bgr, cv2.COLOR_BGR2GRAY)

        return grayscale

    def step(self, action):
            
            # Setting up the initial obs variables
            observation, reward, truncated, info = "", 0, False, "" 

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_TAB:
                        self.gun_type = "single" if self.gun_type == "shotgun" else "shotgun"
                        print(f"Switched to {self.gun_type} mode")
                    elif event.key == pygame.K_SPACE:
                        if self.gun_type == "single":
                            self.fire_single_bullet()
                        else:
                            self.fire_shotgun_bullet()
                    elif event.key == pygame.K_ESCAPE:
                        self.toggle_pause()

            if self.paused:
                return  # Skip the rest of the game loop if paused

            player_moved = False

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
                    bullet = get_collision(zombie.rect, self.bullets)  # Get the bullet causing the collision
                    self.player.score += 1
                    reward += 1
                    self.bullets.remove(bullet)

                    if self.sound:
                        self.zombie_hit.play()

                    # 20% chance to drop a heart
                    if random.randint(1, 100) <= 20:
                        # Drop the heart exactly where the zombie was killed
                        self.health_drop = HealthDrop(zombie.rect.x, zombie.rect.y)
                        print(f"Heart dropped at ({zombie.rect.x}, {zombie.rect.y}) from zombie!")
                
                elif check_collision(zombie.rect, [self.player.rect]):
                    self.player.health -= 1
                    reward -= 1
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

            # Draw the health drop if it exists
            if self.health_drop:
                self.health_drop.draw(self.screen, camera_x, camera_y)


            # Draw the world boundaries for testing
            pygame.draw.rect(self.screen, self.border_color, (0 - camera_x, 0 - camera_y, self.world_width, self.world_height), 5)

            for wall in self.walls:
                pygame.draw.rect(self.screen, self.wall_color, (wall.x - camera_x, wall.y - camera_y, wall.width, wall.height))

            # Draw the treasure chest
            if self.treasure_chest:
                self.treasure_chest.draw(self.screen, camera_x, camera_y)

            # Check for chest opening and health drop collection
            if self.treasure_chest and self.player.rect.colliderect(self.treasure_chest.rect):
                if not self.treasure_chest.is_opened:
                    self.treasure_chest.is_opened = True
                    reward += 1 # Reward for shotgun shell pickup. 

                    # Add shotgun ammo when chest is opened
                    self.shotgun_ammo = min(self.shotgun_ammo + 5, 20)  # Max ammo is 20
                    print(f"Ammo refilled! Current ammo: {self.shotgun_ammo}")

            if self.health_drop and self.player.rect.colliderect(self.health_drop.rect):
                self.player.health = min(self.player.health + 1, 100)  # Increase health by 1 points
                reward += 1 # Reward for health pickup
                print("Heart collected! Health increased.")
                self.health_drop = None  # Remove the heart

            # Update the display
            pygame.display.flip()

            if self.player.health <= 0:
                self.game_over()

            # Cap the frame rate
            self.clock.tick(self.fps)

            if(self.level_goal <= self.player.score):
                self.start_next_level()

            return self._get_obs(), reward, self.done, truncated, self._get_info()