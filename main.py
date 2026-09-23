import pygame
from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
from assets import AssetManager
from game import FlappyGame


def main():
    pygame.init()
    pygame.font.init()
    pygame.mixer.init()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird")
    clock = pygame.time.Clock()

    assets = AssetManager()
    game = FlappyGame(screen, clock, assets)

    running = True
    while running:
        clock.tick(FPS)
        running = game.handle_events()
        game.update()
        game.draw()

    pygame.quit()


if __name__ == '__main__':
    main()