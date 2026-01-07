import pygame
import random
import math
import torch
import sys

# Curriculum learning: Progressive maze difficulty
# Level 0: Empty (just borders)
MAZE_EMPTY = [
    "1111111111111111",
    "1000000000000001",
    "1000000000000001",
    "1000000000000001",
    "1000000000000001",
    "1000000000000001",
    "1000000000000001",
    "1000000000000001",
    "1000000000000001",
    "1000000000000001",
    "1000000000000001",
    "1000000000000001",
    "1000000000000001",
    "1000000000000001",
    "1000000000000001",
    "1111111111111111",
]

# Level 1: Simple obstacles
MAZE_SIMPLE = [
    "1111111111111111",
    "1000000000000001",
    "1000001100000001",
    "1000000000000001",
    "1000000000000001",
    "1000000110000001",
    "1000000000000001",
    "1000000000000001",
    "1000011000000001",
    "1000000000000001",
    "1000000000000001",
    "1000000000110001",
    "1000000000000001",
    "1000000000000001",
    "1000000000000001",
    "1111111111111111",
]

# Level 2: Medium complexity
MAZE_MEDIUM = [
    "1111111111111111",
    "1000000000000001",
    "1011100111100001",
    "1010000000000001",
    "1010000000000001",
    "1000011100000001",
    "1000000000011101",
    "1000000000000001",
    "1011100000000001",
    "1000000000000001",
    "1000000111100001",
    "1000000000000001",
    "1001110000011001",
    "1000000000000001",
    "1000000000000001",
    "1111111111111111",
]

# Level 3: Full complexity (original)
LABYRINTH_16x16 = [
    "1111111111111111",
    "1000000000000001",
    "1011110111110101",
    "1010000000000101",
    "1010111101110101",
    "1000100000000001",
    "1011101111011101",
    "1000000000000001",
    "1011111011111101",
    "1000000000000001",
    "1011111110111101",
    "1010000000100001",
    "1010111001101101",
    "1000100000000101",
    "1000000000000001",
    "1111111111111111",
]

# definition of the Snake class
# Take X and Y position and size as parameters
class Snake:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size

    # define a draw method that takes a window and color as parameters
    def draw(self, window, color):
        # Use pygame.draw.rect to draw a rectangle representing the snake segment
        pygame.draw.rect(window, color, [self.x, self.y, self.size, self.size])

# definition of the Food class
# Take X and Y position, color, and size as parameters
class Food:
    def __init__(self, x, y, color, size):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
    # define a draw method that takes a window object parameter
    def draw(self, window):
        # Use pygame.draw.rect to draw a rectangle representing the food
        pygame.draw.rect(window, self.color, [self.x, self.y, self.size, self.size])

# definition of the game environment class
class game():
    def __init__(self):
        pygame.init()
        # size of the grid
        self.size = 16

        # set the window and title
        self.window_width = 40*self.size
        self.window_height = 40*self.size
        self.window = pygame.display.set_mode((self.window_width, self.window_height),pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.window.fill((255, 255, 255))  # white back ground
        pygame.display.set_caption("Snake Game")

        # wall_cells includes border +  labyrinth walls
        # Start with empty maze for curriculum learning
        self.current_maze = LABYRINTH_16x16
        self.load_labyrinth(self.current_maze)

        # dynamic bricks and counters
        # self.brick_positions = set()
        # self.bricks_count = 1
        # self.food_eaten = 0

        # set colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.blue = (0,0,255)

        # set the size of blocks
        self.snake_size = 40
        self.food_size = 40

        # set the initial position and direction of the snake
        self.snake_x = self.size // 2 * 40 + 40
        self.snake_y = self.size // 2 * 40 + 40
        self.snake_dx = 0
        self.snake_dy = 0

        # set the initial length and body of the snake
        self.snake_length = 2
        self.snake_body = []
        snake = Snake(self.snake_x, self.snake_y, self.snake_size)
        self.snake_body.append(snake)
        snake2 = Snake(self.snake_x-self.snake_size, self.snake_y, self.snake_size)
        self.snake_body.append(snake2)

        # set the initial position of the food while avoiding walls
        while True:
            fx = random.randint(0, self.size-1) * 40
            fy = random.randint(0, self.size-1) * 40
            if (fx, fy) not in self.wall_cells:
                self.food_x = fx
                self.food_y = fy
                break

        # initialize game over flag and score
        self.game_over = False
        self.score = 0
        self.isfps = False
        self.clock = pygame.time.Clock()

        # set font and font size for rendering text
        self.font = pygame.font.SysFont("arial", 32)

        # create a food object
        self.food_ = Food(self.food_x, self.food_y, self.red, self.food_size)

        # set initial observation parameters
        self.reward = 0
        self.n_act = 4
        self.n_obs = 26
        self.life = 256
        self.last_action = 4
        self.distance = math.sqrt(((self.snake_body[0].x-self.food_x)/40)**2+((self.snake_body[0].y-self.food_y)/40)**2)
        self.base_distance = 3
        self.len_num = 0
        self.step_num = 0

        # set the game's frame rate
        self.fps = 20

        # punishment and reward parameters
        # tiny punishment for each step to encourage shorter paths
        self.punish_no_food = 0.001
        # time without food counter
        self.no_food_time = 0
        # max time without food before punishment (increased for labyrinth navigation)
        self.no_food_time_max = 12
        # punishment for hitting itself
        self.punish_byself = 2.2
        # punishment for hitting the wall
        self.punish_bywall = 2.5
        # reward for eating food (dramatically increased for initial learning)
        self.reward_byfood = 3
        # reward for getting closer to food (old variable name)
        self.punish_byfoodclose = 0.05
        # reward for staying alive each step
        self.reward_byalive = 0.01

    # take a step in the environment
    def step(self,action):
        self.len_num += 1
        self.step_num += 1

        # ---------------------- movement -----------------------
        # 0--up   1--down   2--left  3--right
        # update the direction of the snake based on the action
        # prevent the snake from reversing directly
        if action == 0:
            if self.last_action == 1:
                self.self_reward()
                self.game_over = True
            self.snake_dx = 0
            self.snake_dy = -self.snake_size
            self.len_num = 0
        elif action == 1:
            if self.last_action == 0:
                self.self_reward()
                self.game_over = True
            self.snake_dx = 0
            self.snake_dy = self.snake_size
            self.len_num = 0
        elif action == 2:
            if self.last_action == 3:
                self.self_reward()
                self.game_over = True
            self.snake_dx = -self.snake_size
            self.snake_dy = 0
            self.len_num = 0
        elif action == 3:
            if self.last_action == 2:
                self.self_reward()
                self.game_over = True
            self.snake_dx = self.snake_size
            self.snake_dy = 0
            self.len_num = 0
        self.last_action = action
        # ------------------- end of movement -------------------

        self.life -= 1  # decrease life each step, max 256, this is to prevent endless loop
        done = False # signals whether the episode is over

        # check for wall collisions
        if (self.snake_x, self.snake_y) in getattr(self, 'wall_cells', set()):
            self.wall_reward()
            self.game_over = True
        else:
            # update the position of the snake
            self.snake_x += self.snake_dx
            self.snake_y += self.snake_dy

        # create a new snake object for the new head position
        new_head = Snake(self.snake_x, self.snake_y,self.snake_size)
        # insert the new snake head at the beginning of the snake body list
        self.snake_body.insert(0, new_head)
        # remove the last segment of the snake body if the length exceeds the snake length
        if len(self.snake_body) > self.snake_length:
            del self.snake_body[-1]
        # check for self-collisions
        for segment in self.snake_body[1:]:
            # give a penalty if the head collides with any segment of the body
            if segment.x == new_head.x and segment.y == new_head.y:
                self.self_reward()
                self.game_over = True

        # calculate rewards only if the game is not over
        if not self.game_over:

            self.reward += self.reward_byalive

            # WARNING: punish for moving toward a wall
            self.check_wall_ahead()

            # step reward and punishments
            self.step_reward()

            # check if the snake has eaten the food
            self.iseat(new_head)

        reward = self.reward
        self.reward = 0
        if self.life <= 0 or self.game_over:
            done = True
        return self.obs_(), reward, done, self.snake_length

    # calculate a linear gradient between colors for snake body segments
    def linear_gradient(self,start_color, end_color,length):
        r_step = (end_color[0] - start_color[0]) / 40
        g_step = (end_color[1] - start_color[1]) / 40
        b_step = (end_color[2] - start_color[2]) / 40
        # save the gradient
        gradient = []
        # calculate the step size for each color channel, and save to list
        for i in range(length):
            r = int(start_color[0] + r_step * i)
            g = int(start_color[1] + g_step * i)
            b = int(start_color[2] + b_step * i)
            gradient.append((r, g, b))
        return gradient
    
    # render the game environment
    def render(self,isdraw,episodes):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        # fill the window with white color
        self.window.fill(self.white)
        # draw walls
        wall_color = (100,100,100)
        for (wx, wy) in getattr(self, 'wall_cells', set()):
            pygame.draw.rect(self.window, wall_color, (wx, wy, self.snake_size, self.snake_size))
        if not isdraw:
            pygame.display.update()
            return
        for i, segment in enumerate(self.snake_body):
            # calculate the color for each segment based on its position in the body
            segment_color = self.linear_gradient(self.blue, self.white, self.snake_length)
            # if it's the head of the snake, draw it in green
            if i == 0:
                pygame.draw.rect(self.window, self.green,
                                 (segment.x, segment.y, self.snake_size, self.snake_size))
            else:
                # if it's not the head, still draw a rectangle
                segment.draw(self.window,segment_color[i-1])
        # call the food object's draw method, passing the window object as a parameter
        self.food_.draw(self.window)
        # use the font object to render the score text, color is white
        score_text = self.font.render("Score: " + str(self.score), True, self.black)
        # draw the score text at the top-left corner of the window
        self.window.blit(score_text, (0, 0))
        # use the font object to render the episodes text, color is white
        episodes_text = self.font.render("episodes: " + str(episodes), True, self.black)
        # get the size of the episodes text to position it correctly at the top-right corner
        episodes_text_width, episodes_text_height = self.font.size("episodes: " + str(episodes))
        # draw the episodes text at the top-right corner of the window
        self.window.blit(episodes_text, (self.window.get_width() - episodes_text_width, 0))
        # update the window display
        pygame.display.update()
        # set the clock object's delay time to control the game's frame rate
        self.clock.tick(self.fps)
    
    def check_wall_ahead(self):
        """
        WARNING system: Punish agent for moving TOWARD imminent collision.
        Only warns about NEXT cell (1 step ahead) - strong and consistent.
        
        This allows corridor navigation (can move toward distant walls while planning to turn)
        while preventing blind collisions (strong warning when wall is immediate).
        """
        x, y = self.snake_x, self.snake_y
        
        # Check ONLY the next cell in movement direction
        # Warning strong enough to matter but not overwhelm pathfinding
        warning_strength = 0.2  # Balanced: allows learning while preventing collisions
        
        # 0=up, 1=down, 2=left, 3=right
        if self.last_action == 0:  # Moving up
            check_y = y - 40
            if check_y < 0 or (x, check_y) in self.wall_cells:
                self.reward -= warning_strength
        elif self.last_action == 1:  # Moving down
            check_y = y + 40
            if check_y >= self.size * 40 or (x, check_y) in self.wall_cells:
                self.reward -= warning_strength
        elif self.last_action == 2:  # Moving left
            check_x = x - 40
            if check_x < 0 or (check_x, y) in self.wall_cells:
                self.reward -= warning_strength
        elif self.last_action == 3:  # Moving right
            check_x = x + 40
            if check_x >= self.size * 40 or (check_x, y) in self.wall_cells:
                self.reward -= warning_strength

    def load_labyrinth(self, maze):
        self.wall_cells = set()
        for r, row in enumerate(maze):
            for c, cell in enumerate(row):
                if cell == "1" or cell == 1:
                    # print("wall at:", r, c)
                    self.wall_cells.add((c * 40, r * 40))
    
    def set_difficulty_level(self, level):
        """
        Change maze difficulty based on training progress.
        Level 0: Empty (just borders) - for initial learning
        Level 1: Simple obstacles - basic wall avoidance
        Level 2: Medium complexity - corridor navigation  
        Level 3: Full labyrinth - final challenge
        
        Key principle: Keep rewards CONSTANT to maintain learning signal.
        Only adjust punishments to teach wall avoidance gradually.
        """
        mazes = [MAZE_EMPTY, MAZE_SIMPLE, MAZE_MEDIUM, LABYRINTH_16x16]
        if 0 <= level < len(mazes):
            self.current_maze = mazes[level]
            self.load_labyrinth(self.current_maze)
            
            # Only gradually increase collision punishment
            if level == 0:
                self.punish_bywall = 2.5
                self.reward_byfood = 3
            elif level == 1:
                self.punish_bywall = 3
                self.reward_byfood = 3.6
            elif level == 2:
                self.punish_bywall = 4
                self.reward_byfood = 4.8
            elif level == 3:
                self.punish_bywall = 5
                self.reward_byfood = 6
            
            print(f"\n{'='*60}")
            print(f"Maze difficulty increased to Level {level}")
            print(f"Wall collision punishment: {self.punish_bywall}")
            print(f"Food reward: {self.reward_byfood}")
            print(f"{'='*60}\n")


    def iseat(self,snake):
        # check if the snake head collides with the food
        if snake.x == self.food_.x and snake.y == self.food_.y:
            self.eat_reward()
            food_born = False
            while not food_born:
                # randomly generate new food position
                self.food_.x = random.randint(0, self.size - 1) * 40
                self.food_.y = random.randint(0, self.size - 1) * 40

                # if the new food position coincides with any segment of the snake body, regenerate
                num = 0
                for segment in self.snake_body:
                    if segment.x == self.food_.x and segment.y == self.food_.y:
                        num += 1
                        continue

                # ensure food not surrounded by walls
                # if self.food_surrounded_by_walls() >= 3:
                #     continue

                # also ensure food not spawned inside a wall
                if num == 0 and (self.food_.x, self.food_.y) not in getattr(self, 'wall_cells', set()):
                    food_born = True

            # increase snake length
            self.snake_length += 1

            # increase score
            self.score += 1
            self.life = 200
            self.no_food_time = 0

            # track food eaten and spawn bricks every 2 fruits (max 10 bricks)
            # self.food_eaten = getattr(self, 'food_eaten', 0) + 1
            # if self.food_eaten % 2 == 0:
            #     current = len(getattr(self, 'brick_positions', set()))
            #     if current < 20:
            #         placed = False
            #         while not placed:
            #             # generate random position for brick
            #             r = random.randint(1, self.size-2)
            #             c = random.randint(1, self.size-2)
            #             pos = (c*40, r*40)

            #             # check distance from snake head, cannot be too close
            #             head = (self.snake_body[0].x, self.snake_body[0].y)
            #             distance = abs(pos[0]-head[0]) + abs(pos[1]-head[1])
            #             if distance < 120:
            #                 continue
            #             if pos in self.wall_cells:
            #                 continue
            #             occupied = False
            #             for seg in self.snake_body:
            #                 if (seg.x, seg.y) == pos:
            #                     occupied = True
            #                     break
            #             if pos == (self.food_.x, self.food_.y):
            #                 occupied = True
            #             if occupied:
            #                 continue
            #             self.brick_positions.add(pos)
            #             self.wall_cells.add(pos)
            #             placed = True
        else:
            self.no_food_time += 1
            self.long_time_noeat_reward()

    # reward for eating food
    def eat_reward(self):
        # Large reward for eating food (constant across all difficulty levels)
        if self.snake_length <= 5:
            self.reward += self.reward_byfood
        else:
            self.reward += self.snake_length * self.reward_byfood

    # penalty for hitting a wall
    def wall_reward(self):
        # Reduced punishment for early learning to encourage exploration
        self.reward -= self.punish_bywall

    # penalty for hitting itself
    def self_reward(self):
        # Constant punishment for self-collision
        self.reward -= self.punish_byself

    # small reward/punishment for each step taken
    def step_reward(self):
        # Adaptive distance metric based on maze complexity
        # Euclidean for empty maze (direct paths possible)
        # Manhattan for mazes with walls (must navigate around obstacles)
        if self.current_maze == MAZE_EMPTY:
            # Euclidean distance - optimal for open space
            distance = math.sqrt(((self.snake_body[0].x-self.food_x)/40)**2+((self.snake_body[0].y-self.food_y)/40)**2)
        else:
            # Manhattan distance - better for navigating around walls
            distance = abs(self.snake_body[0].x - self.food_x)/40 + abs(self.snake_body[0].y - self.food_y)/40
        
        self.reward += ((self.distance-distance)/self.base_distance)*self.punish_byfoodclose
        self.distance = distance

    # penalty for not eating food for a long time
    def long_time_noeat_reward(self):
        if self.no_food_time > self.no_food_time_max:
            self.reward -= self.punish_no_food

    # check if the snake is going straight for too long
    # def isstraight(self):
    #     if self.len_num >= self.len_num_max:
    #         self.reward -= self.punish_step


    def reset(self):
        # set the size of the snake and food
        self.snake_size = 40
        self.food_size = 40
        # set the initial position and direction of the snake
        self.snake_x = self.size // 2 * 40 + 40
        self.snake_y = self.size // 2 * 40 + 40
        self.snake_dx = 0
        self.snake_dy = 0
        # set the initial length and body list of the snake
        self.snake_length = 2
        self.snake_body = []
        snake = Snake(self.snake_x, self.snake_y,self.snake_size)
        self.snake_body.append(snake)
        snake2 = Snake(self.snake_x - self.snake_size, self.snake_y,self.snake_size)
        self.snake_body.append(snake2)
        # reset walls and labyrinth
        self.load_labyrinth(self.current_maze)

        # self.brick_positions = set()
        # self.bricks_count = 1
        # self.food_eaten = 0
        # set the initial position of the food (avoid walls)
        while True:
            fx = random.randint(0, self.size-1) * 40
            fy = random.randint(0, self.size-1) * 40
            if (fx, fy) not in self.wall_cells:
                self.food_x = fx
                self.food_y = fy
                break
        # set the initial game state and score
        self.game_over = False
        self.score = 0
        # create a food object with the food's position, color, and size parameters
        self.food_ = Food(self.food_x, self.food_y, self.red, self.food_size)
        # place initial brick
        # placed = 0
        # while placed < self.bricks_count:
        #     r = random.randint(1, self.size-2)
        #     c = random.randint(1, self.size-2)
        #     pos = (c*40, r*40)
        #     if pos in self.wall_cells:
        #         continue
        #     occupied = False
        #     for seg in self.snake_body:
        #         if (seg.x, seg.y) == pos:
        #             occupied = True
        #             break
        #     if pos == (self.food_x, self.food_y):
        #         occupied = True
        #     if occupied:
        #         continue
        #     self.brick_positions.add(pos)
        #     self.wall_cells.add(pos)
        #     placed += 1
        # create rewards and observation parameters
        self.reward = 0
        self.life = 256
        self.last_action = 4
        self.len_num = 0
        self.step_num = 0
        self.distance = math.sqrt(((self.snake_body[0].x-self.food_x)/40)**2+((self.snake_body[0].y-self.food_y)/40)**2)
        return self.obs_()

    def obs_(self):
        # relative x and y coordinates between the snake's head and the food
        # whether there is the snake's own body or game boundaries 
        # above, below, left, or right of the snake's head as state
        # and put them into a tensor
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        obs = torch.zeros(1, self.n_obs, device=device)
        # relative x and y coordinates between the snake's head and the food
        obs[0][0] = (self.snake_body[0].x - self.food_.x)/40
        obs[0][1] = (self.snake_body[0].y - self.food_.y)/40  # Fixed: was using .x instead of .y
        
        # whether there is the snake's own body above, below, left, or right of the snake's head, and if so, how far
        # Initialize all to -1 (no body detected)
        obs[0][2] = -1  # up
        obs[0][3] = -1  # down
        obs[0][4] = -1  # left
        obs[0][5] = -1  # right
        
        # Find NEAREST body segment in each direction
        for segment in self.snake_body[1:]:
            # up
            if self.snake_body[0].x == segment.x and self.snake_body[0].y - segment.y > 0:
                dist = (self.snake_body[0].y - segment.y)/40 - 1
                if obs[0][2] == -1 or dist < obs[0][2]:
                    obs[0][2] = dist
            # down
            if self.snake_body[0].x == segment.x and self.snake_body[0].y - segment.y < 0:
                dist = (segment.y - self.snake_body[0].y)/40 - 1
                if obs[0][3] == -1 or dist < obs[0][3]:
                    obs[0][3] = dist
            # left
            if self.snake_body[0].y == segment.y and self.snake_body[0].x - segment.x > 0:
                dist = (self.snake_body[0].x - segment.x)/40 - 1
                if obs[0][4] == -1 or dist < obs[0][4]:
                    obs[0][4] = dist
            # right
            if self.snake_body[0].y == segment.y and self.snake_body[0].x - segment.x < 0:
                dist = (segment.x - self.snake_body[0].x)/40 - 1
                if obs[0][5] == -1 or dist < obs[0][5]:
                    obs[0][5] = dist

        # direction of the snake's head
        if self.snake_dx == 0 and self.snake_dy < 0:
            obs[0][6] = 1
        elif self.snake_dx == 0 and self.snake_dy > 0:
            obs[0][6] = 2
        elif self.snake_dx < 0 and self.snake_dy == 0:
            obs[0][6] = 3
        elif self.snake_dx > 0 and self.snake_dy == 0:
            obs[0][6] = 4
        # whether the snake's head is at the boundary
        max_coord = (self.size - 1) * 40
        obs[0][7] = self.snake_x == 0 or self.snake_x == max_coord or self.snake_y == 0 or self.snake_y == max_coord
        # distance from the snake's head to the four boundaries
        obs[0][8] = self.snake_body[0].x/40
        obs[0][9] = self.snake_body[0].y/40
        obs[0][10] = (self.size*40-40-self.snake_body[0].x)/40
        obs[0][11] = (self.size*40-40-self.snake_body[0].y)/40
        # whether the food is at the boundary
        obs[0][12] = self.food_.x == 0 or self.food_.x == max_coord or self.food_.y == 0 or self.food_.y == max_coord
        # distance from the food to the four boundaries
        obs[0][13] = self.food_.x/40
        obs[0][14] = self.food_.y/40
        obs[0][15] = (self.size*40-40-self.food_.x)/40
        obs[0][16] = (self.size*40-40-self.food_.y)/40
        # whether there is an obstacle between the snake's head and the food
        if self.snake_body[0].x == self.food_.x:
            for segment in self.snake_body[1:]:
                if segment.y > min(self.snake_body[0].y,self.food_.y) and segment.y < max(self.snake_body[0].y,self.food_.y):
                    obs[0][17] = 1
                    # relative distance between the snake's head and this obstacle
                    obs[0][18] = math.fabs((self.snake_body[0].y - segment.y)/40) - 1
                    # relative distance between the food and this obstacle
                    obs[0][19] = math.fabs((self.food_.y - segment.y)/40) - 1
                    break
        elif self.snake_body[0].y == self.food_.y:
            for segment in self.snake_body[1:]:
                if segment.x > min(self.snake_body[0].x,self.food_.x) and segment.x < max(self.snake_body[0].x,self.food_.x):
                    obs[0][17] = 1
                    # relative distance between the snake's head and this obstacle
                    obs[0][18] = math.fabs((self.snake_body[0].x - segment.x)/40) - 1
                    # relative distance between the food and this obstacle
                    obs[0][19] = math.fabs((self.food_.x - segment.x)/40) - 1
                    break
        else:
            obs[0][17] = 0
            obs[0][18] = -1
            obs[0][19] = -1
        
        # Distance to nearest wall in each direction (up, down, left, right)
        # Up
        obs[0][20] = -1
        for dist in range(1, self.size):
            check_y = self.snake_body[0].y - dist * 40
            if check_y < 0 or (self.snake_body[0].x, check_y) in self.wall_cells:
                obs[0][20] = dist - 1
                break
        
        # Down
        obs[0][21] = -1
        for dist in range(1, self.size):
            check_y = self.snake_body[0].y + dist * 40
            if check_y >= self.size * 40 or (self.snake_body[0].x, check_y) in self.wall_cells:
                obs[0][21] = dist - 1
                break
        
        # Left
        obs[0][22] = -1
        for dist in range(1, self.size):
            check_x = self.snake_body[0].x - dist * 40
            if check_x < 0 or (check_x, self.snake_body[0].y) in self.wall_cells:
                obs[0][22] = dist - 1
                break
        
        # Right
        obs[0][23] = -1
        for dist in range(1, self.size):
            check_x = self.snake_body[0].x + dist * 40
            if check_x >= self.size * 40 or (check_x, self.snake_body[0].y) in self.wall_cells:
                obs[0][23] = dist - 1
                break
        
        # PERPENDICULAR wall detection (relative to movement direction)
        # Critical for learning "don't turn into a wall"
        x, y = self.snake_body[0].x, self.snake_body[0].y
        
        # obs[24] = wall immediately to the LEFT (perpendicular to movement)
        # obs[25] = wall immediately to the RIGHT (perpendicular to movement)
        obs[0][24] = 0  # 0 = no wall, 1 = wall present
        obs[0][25] = 0
        
        # Direction: 0=up, 1=down, 2=left, 3=right
        current_dir = int(obs[0][6].item()) if obs[0][6] != 0 else self.last_action
        
        if current_dir == 0:  # Moving up
            # Left = west, Right = east
            if (x - 40, y) in self.wall_cells or x - 40 < 0:
                obs[0][24] = 1
            if (x + 40, y) in self.wall_cells or x + 40 >= self.size * 40:
                obs[0][25] = 1
        elif current_dir == 1:  # Moving down
            # Left = east, Right = west
            if (x + 40, y) in self.wall_cells or x + 40 >= self.size * 40:
                obs[0][24] = 1
            if (x - 40, y) in self.wall_cells or x - 40 < 0:
                obs[0][25] = 1
        elif current_dir == 2:  # Moving left
            # Left = south, Right = north
            if (x, y + 40) in self.wall_cells or y + 40 >= self.size * 40:
                obs[0][24] = 1
            if (x, y - 40) in self.wall_cells or y - 40 < 0:
                obs[0][25] = 1
        elif current_dir == 3:  # Moving right
            # Left = north, Right = south
            if (x, y - 40) in self.wall_cells or y - 40 < 0:
                obs[0][24] = 1
            if (x, y + 40) in self.wall_cells or y + 40 >= self.size * 40:
                obs[0][25] = 1
        
        return obs