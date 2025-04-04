from matplotlib import animation
import matplotlib.pyplot as plt


def visualize(env, ax=None, timestep=0, speed=1):
    """
    Visualizes vehicle positions on the road.

    If `ax` is provided, it updates an existing plot (for animation).
    Otherwise, it creates a new figure and axis.

    Parameters:
    - env: Environment object containing road and vehicle data
    - ax: Matplotlib axis object (for animation)
    - timestep: Current timestep (for display)
    - speed: Speed multiplier for animation
    """
    # Create a new figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 5))
    else:
        ax.clear()  # Clear previous frame for animation

    # Set plot limits
    ax.set_xlim(0, env.road_length)
    ax.set_ylim(0, env.road_width)

    # Add title and labels
    ax.set_title("Vehicle Positions on Road")
    ax.set_xlabel("Distance along the road (m)")
    ax.set_ylabel("Lane Position (m)")

    # Plot vehicle positions
    ax.scatter(
        env.positions[:, 0],
        env.positions[:, 1],
        c="red",
        marker="s",
        s=50,
        label="Vehicles",
    )

    # Plot edge positions
    ax.scatter(
        env.edge_positions[:, 0],
        env.edge_positions[:, 1],
        c="blue",
        marker="o",
        s=500,
        label="Edges",
    )

    # Draw lane dividers
    for lane in range(env.num_lanes):
        lane_y = (lane + 0.5) * (env.road_width / env.num_lanes)
        ax.hlines(
            lane_y,
            0,
            env.road_length,
            colors="r",
            linestyles="dashed",
            label="Lane Divider" if lane == 0 else "",
        )

    # Add timestep annotation
    ax.annotate(
        f"Timestep: {timestep / 10:.1f} s (x{speed})",
        xy=(1, 1.05),
        xycoords="axes fraction",
        fontsize=12,
        color="black",
        bbox=dict(facecolor="white", alpha=0.7),
        ha="right",
    )

    # Return figure and axis if created
    if ax is None:
        return fig, ax


def render(env, timesteps, speed=1):
    """
    Renders an animation of the environment over a series of timesteps.

    Parameters:
    - env: Environment object containing road and vehicle data
    - timesteps: Number of timesteps to animate
    - speed: Speed multiplier for animation
    """
    # Create a figure and axis for the animation
    fig, ax = plt.subplots(figsize=(20, 5))

    def update(frame):
        """
        Update function for each frame of the animation.
        """
        env.step(None)  # Update environment state
        visualize(env, ax=ax, timestep=frame, speed=speed)

    # Create the animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=timesteps,
        interval=100 / speed,  # Adjust interval based on speed
    )

    # Display the animation
    plt.show()
