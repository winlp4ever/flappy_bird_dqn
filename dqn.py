from ple.games.flappybird import FlappyBird
from ple import PLE


game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)
p.init()

agent = myAgentHere(allowed_actions=p.getActionSet())


nb_frames = 1000
reward = 0.0

for i in range(nb_frames):
    if p.game_over():
        p.reset_game()

    observation = p.getScreenRGB()
    action = agent.pickAction(reward, observation)
    reward = p.act(action)