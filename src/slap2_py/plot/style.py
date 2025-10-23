import matplotlib.pyplot as plt
import os


def slap_style(version="im"):
    if version == "im":
        style_path = os.path.join(
            os.path.dirname(__file__), "style", "slap_im.mplstyle"
        )
        return plt.style.use(style_path)
    elif version == "fig":
        style_path = os.path.join(
            os.path.dirname(__file__), "style", "slap_fig.mplstyle"
        )
        return plt.style.use(style_path)
    else:
        raise ValueError(f"Invalid version: {version}")
