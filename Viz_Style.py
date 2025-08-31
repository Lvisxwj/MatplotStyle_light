from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import numpy as np
import sys

import warnings
# 忽略所有 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)



def in_notebook():
    """Check if running inside Jupyter Notebook"""
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except:
        return False
    return True


class XWJ_Style:
    """
    XWJ_Style
    =========
    - 从 YAML 加载配置，统一 matplotlib / seaborn 风格
    - 支持 lineplot, scatterplot, barplot, hist, fill_between, heatmap,
      boxplot, violinplot, pie, polarplot, radarchart, colorbar 等
    - grid / legend / baseline 开关由 main 决定
    - 属性（linewidth, alpha, linestyle 等）全部由 YAML 控制
    """

    # ============================
    # 初始化 & 配置加载
    # ============================
    def __init__(self, yaml_path="style_config.yaml"):
        self.yaml_path = yaml_path
        self.config = self._load_config()
        self.color_cycle = list(self.config["colors"].values())
        self.color_index = 0
        self._apply_global_style()

    def _load_config(self):
        """加载 YAML 配置"""
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_next_color(self):
        """颜色循环：每次调用取下一个颜色"""
        color = self.color_cycle[self.color_index % len(self.color_cycle)]
        self.color_index += 1
        return color

    def reset_color_cycle(self):
        """每次新图时重置颜色索引，保证统一"""
        self.color_index = 0

# ==================I Just Dont Fucking Get IT========================

    def __apply_global_style(self):
        """版本1：严格用YAML配置"""
        font = self.config.get("font", {})
        plt.rcParams['font.sans-serif'] = [font.get("Chinese", "SimHei")] + font.get("fallback", [])
        plt.rcParams['axes.unicode_minus'] = False
        style = "whitegrid" if self.config.get("grid", {}).get("visible", True) else "white"
        sns.set_style(style)

    def _apply_global_style(self):
        """版本2：自定义强制字体"""
        font_list = ["SimHei", "centurygothic_bold", "Times New Roman"]
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': font_list,
            'axes.unicode_minus': False,
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'axes.titleweight': 'bold',
            'axes.labelweight': 'normal'
        })
        grid_visible = self.config.get("grid", {}).get("visible", True)
        sns.set_style("whitegrid" if grid_visible else "white")
        sns.set_context("notebook")

    def ___apply_global_style(self):
        """版本3：最简配置"""
        font = self.config.get("font", {})
        plt.rcParams['font.sans-serif'] = [font.get("family", "SimHei")]
        plt.rcParams['axes.unicode_minus'] = False
        style = "whitegrid" if self.config.get("grid", {}).get("visible", True) else "white"
        sns.set_style(style)

    def __apply_global_style(self):
        """全局字体、风格设置"""
        font_cfg = self.config.get("font", {})
        plt.rcParams["font.family"] = font_cfg.get("family", "SimHei")
        plt.rcParams["font.size"] = font_cfg.get("size", 12)
        plt.rcParams["axes.titlesize"] = font_cfg.get("title_size", 14)
        plt.rcParams["axes.titleweight"] = font_cfg.get("title_weight", "bold")
        plt.rcParams["axes.labelweight"] = font_cfg.get("weight", "bold")
        plt.rcParams["axes.unicode_minus"] = False
        sns.set(style="whitegrid", font=font_cfg.get("family", "SimHei"))

# ==================I Just Dont Fucking Get IT========================


    def _apply_grid(self, show_grid=False):
        """应用网格样式（是否显示由 main 控制）"""
        if show_grid:
            grid_cfg = self.config.get("grid", {})
            plt.grid(
                True,
                alpha=grid_cfg.get("alpha", 0.3),
                linewidth=grid_cfg.get("linewidth", 0.5),
                linestyle=grid_cfg.get("linestyle", "-"),
            )
        else:
            plt.grid(False)

    def _apply_legend(self, add_legend=False, **kwargs):
        """应用图例样式（是否显示由 main 控制）"""
        if add_legend:
            legend_cfg = self.config.get("legend", {})
            plt.legend(
                title=kwargs.get("title", legend_cfg.get("title", "")),
                loc=legend_cfg.get("loc", "best"),
                frameon=legend_cfg.get("frameon", True),
                fancybox=legend_cfg.get("fancybox", True),
                shadow=legend_cfg.get("shadow", True),
                title_fontsize=legend_cfg.get("title_fontsize", 11),
                fontsize=legend_cfg.get("fontsize", 12),
            )

    def _apply_baseline(self, show_baseline=False):
        """基准线样式（是否显示由 main 控制）"""
        if show_baseline:
            base_cfg = self.config.get("baseline", {})
            plt.axhline(
                y=0,
                color=self.get_next_color(),
                alpha=base_cfg.get("alpha", 0.5),
                linewidth=base_cfg.get("linewidth", 1),
                linestyle=base_cfg.get("linestyle", "-"),
            )

    def _new_figure(self):
        """新建画布"""
        fig_cfg = self.config.get("figure", {})
        figsize = fig_cfg.get("figsize", [12, 8])
        self.reset_color_cycle()
        return plt.figure(figsize=figsize)

    def subplots(self, nrows=1, ncols=1, figsize=None):
        """新建子图（兼容 fig, ax / fig, (ax1, ax2) / fig, axes）"""
        fig_cfg = self.config.get("figure", {})
        if figsize is None:
            figsize = fig_cfg.get("figsize", [12, 8])
        self.reset_color_cycle()
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        return fig, axes

    def lineplot(self, x, y, label=None, **kwargs):
        cfg = self.config.get("defaults", {})
        plt.plot(
            x,
            y,
            label=label,
            color=self.get_next_color(),
            linewidth=cfg.get("linewidth", 2.5),
            alpha=cfg.get("alpha", 0.9),
            **kwargs,
        )

    def scatterplot(self, x, y, label=None, **kwargs):
        cfg = self.config.get("scatterplot", {})
        edgecolor = self.config["colors"].get(
            cfg.get("edgecolor"), cfg.get("edgecolor", "black")
        )
        plt.scatter(
            x,
            y,
            label=label,
            color=self.get_next_color(),
            s=cfg.get("s", 50),
            alpha=cfg.get("alpha", 0.8),
            edgecolor=edgecolor,
            **kwargs,
        )

    def barplot(self, x, height, label=None, **kwargs):
        plt.bar(
            x,
            height,
            label=label,
            color=self.get_next_color(),
            alpha=self.config["defaults"].get("alpha", 0.9),
            **kwargs,
        )

    def hist(self, data, label=None, **kwargs):
        cfg = self.config.get("hist", {})
        edgecolor = self.config["colors"].get(
            cfg.get("edgecolor"), cfg.get("edgecolor", "black")
        )
        plt.hist(
            data,
            bins=cfg.get("bins", 30),
            alpha=cfg.get("alpha", 0.7),
            color=self.get_next_color(),
            edgecolor=edgecolor,
            label=label,
            **kwargs,
        )

    def fill_between(self, x, y1, y2=0, label=None, **kwargs):
        cfg = self.config.get("fill_between", {})
        plt.fill_between(
            x,
            y1,
            y2,
            color=self.get_next_color(),
            alpha=cfg.get("alpha", 0.3),
            label=label,
            **kwargs,
        )

    def axhline(self, y=0, **kwargs):
        cfg = self.config.get("axline", {})
        plt.axhline(
            y=y,
            color=self.get_next_color(),
            linestyle=cfg.get("linestyle", "--"),
            alpha=cfg.get("alpha", 0.5),
            linewidth=cfg.get("linewidth", 1),
            **kwargs,
        )

    def axvline(self, x=0, **kwargs):
        cfg = self.config.get("axline", {})
        plt.axvline(
            x=x,
            color=self.get_next_color(),
            linestyle=cfg.get("linestyle", "--"),
            alpha=cfg.get("alpha", 0.5),
            linewidth=cfg.get("linewidth", 1),
            **kwargs,
        )


    def heatmap_v2(self, data, **kwargs):
        cfg = self.config.get("heatmap", {})
        # 用自定义 colors 列表生成 colormap
        cmap = ListedColormap(self.color_cycle)

        ax = sns.heatmap(
            data,
            cmap=cmap,  # 用自定义的调色板 丑
            annot=cfg.get("annot", True),
            fmt=cfg.get("fmt", ".1f"),
            linewidths=cfg.get("linewidths", 0.5),
            **kwargs,
        )
        return ax

    def heatmap(self, data, **kwargs):
        cfg = self.config.get("heatmap", {})
        ax = sns.heatmap(
            data,
            cmap=cfg.get("cmap", "YlGnBu"),
            annot=cfg.get("annot", True),
            fmt=cfg.get("fmt", ".1f"),
            linewidths=cfg.get("linewidths", 0.5),
            **kwargs,
        )
        # 返回 heatmap 对象（seaborn 的 AxesSubplot）
        return ax


    def heatmap_v1(self, data, **kwargs):
        cfg = self.config.get("heatmap", {})
        sns.heatmap(
            data,
            cmap=cfg.get("cmap", "YlGnBu"),
            annot=cfg.get("annot", True),
            fmt=cfg.get("fmt", ".1f"),
            linewidths=cfg.get("linewidths", 0.5),
            **kwargs,
        )

    def boxplot(self, data, **kwargs):
        cfg = self.config.get("boxplot", {})
        sns.boxplot(
            data=data,
            linewidth=cfg.get("linewidth", 1.5),
            fliersize=cfg.get("fliersize", 3),
            whis=cfg.get("whis", 1.5),
            notch=cfg.get("notch", False),
            showcaps=cfg.get("showcaps", True),
            palette=self.color_cycle,
            **kwargs,
        )

    def violinplot(self, data, **kwargs):
        cfg = self.config.get("violinplot", {})
        sns.violinplot(
            data=data,
            linewidth=cfg.get("linewidth", 1.2),
            inner=cfg.get("inner", "box"),
            cut=cfg.get("cut", 0),
            scale=cfg.get("scale", "width"),
            bw=cfg.get("bw", 0.2),
            palette=self.color_cycle,
            **kwargs,
        )

    def pie(self, x, labels=None, **kwargs):
        cfg = self.config.get("pie", {})
        explode = cfg.get("explode", [])
        if not explode or len(explode) != len(x):
            explode = [0] * len(x)   # 自动修复，长度必须和数据一致

        plt.pie(
            x,
            labels=labels,
            colors=self.color_cycle[: len(x)],
            startangle=cfg.get("startangle", 90),
            autopct=cfg.get("autopct", "%.1f%%"),
            pctdistance=cfg.get("pctdistance", 0.6),
            labeldistance=cfg.get("labeldistance", 1.1),
            explode=explode,
            shadow=cfg.get("shadow", False),
            radius=cfg.get("radius", 1.0),
            **kwargs,
        )



    def pie_v1(self, x, labels=None, **kwargs):
        cfg = self.config.get("pie", {})
        plt.pie(
            x,
            labels=labels,
            colors=self.color_cycle[: len(x)],
            startangle=cfg.get("startangle", 90),
            autopct=cfg.get("autopct", "%.1f%%"),
            pctdistance=cfg.get("pctdistance", 0.6),
            labeldistance=cfg.get("labeldistance", 1.1),
            explode=cfg.get("explode", []),
            shadow=cfg.get("shadow", False),
            radius=cfg.get("radius", 1.0),
            **kwargs,
        )

    def polarplot_v1(self, theta, r, label=None, **kwargs):
        cfg = self.config.get("polarplot", {})
        ax = plt.subplot(111, polar=True)
        ax.plot(
            theta,
            r,
            label=label,
            color=self.get_next_color(),
            linewidth=cfg.get("linewidth", 2),
            alpha=cfg.get("alpha", 0.9),
            linestyle=cfg.get("linestyle", "-"),
            marker=cfg.get("marker", "o"),
            **kwargs,
        )
        return ax

    def radarchart_v1(self, categories, values, label=None, **kwargs):
        cfg = self.config.get("radarchart", {})
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        values += values[:1]  # 闭合雷达图
        angles += angles[:1]

        ax = plt.subplot(111, polar=True)
        ax.plot(
            angles,
            values,
            label=label,
            color=self.get_next_color(),
            linewidth=cfg.get("linewidth", 2),
            linestyle=cfg.get("linestyle", "-"),
            **kwargs,
        )
        if cfg.get("fill", True):
            ax.fill(
                angles,
                values,
                color=self.get_next_color(),
                alpha=cfg.get("fill_alpha", 0.6),
            )
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        return ax
    
    def polarplot(self, theta, r, label=None, **kwargs):
        cfg = self.config.get("polarplot", {})
        ax = plt.subplot(111, polar=True)
        color = self.get_next_color()
        ax.plot(
            theta,
            r,
            label=label,
            color=color,
            linewidth=cfg.get("linewidth", 2),
            alpha=cfg.get("alpha", 0.9),
            linestyle=cfg.get("linestyle", "-"),
            marker=cfg.get("marker", "o"),
            **kwargs,
        )
        return ax

    def radarchart(self, categories, values, label=None, **kwargs):
        cfg = self.config.get("radarchart", {})
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        values += values[:1]  # 闭合雷达图
        angles += angles[:1]

        ax = plt.subplot(111, polar=True)
        color = self.get_next_color()
        ax.plot(
            angles,
            values,
            label=label,
            color=color,
            linewidth=cfg.get("linewidth", 2),
            linestyle=cfg.get("linestyle", "-"),
            **kwargs,
        )
        if cfg.get("fill", True):
            ax.fill(
                angles,
                values,
                color=color,
                alpha=cfg.get("alpha", 0.6),
            )
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        return ax


    def colorbar(self, mappable=None, **kwargs):
        cfg = self.config.get("colorbar", {})
        if mappable is None:
            mappable = plt.gci()  # 尝试获取当前图像
        if mappable is None:
            raise RuntimeError("No mappable found for colorbar. Please pass mappable explicitly.")
        cbar = plt.colorbar(
            mappable,
            shrink=cfg.get("shrink", 0.8),
            aspect=cfg.get("aspect", 20),
            pad=cfg.get("pad", 0.05),
            **kwargs,
        )
        cbar.ax.tick_params(labelsize=cfg.get("labelsize", 10))
        return cbar



    def colorbar_v2(self, mappable=None, **kwargs):
        cfg = self.config.get("colorbar", {})
        cbar = plt.colorbar(
            mappable,
            shrink=cfg.get("shrink", 0.8),
            aspect=cfg.get("aspect", 20),
            pad=cfg.get("pad", 0.05),
            **kwargs,
        )
        cbar.ax.tick_params(labelsize=cfg.get("labelsize", 10))
        return cbar


    def set_title(self, text, **kwargs):
        font_cfg = self.config.get("font", {})
        plt.title(
            text,
            fontsize=font_cfg.get("title_size", 14),
            fontweight=font_cfg.get("title_weight", "bold"),
            **kwargs,
        )

    def set_xlabel(self, text, **kwargs):
        font_cfg = self.config.get("font", {})
        plt.xlabel(
            text,
            fontsize=font_cfg.get("size", 12),
            fontweight=font_cfg.get("weight", "bold"),
            **kwargs,
        )

    def set_ylabel(self, text, **kwargs):
        font_cfg = self.config.get("font", {})
        plt.ylabel(
            text,
            fontsize=font_cfg.get("size", 12),
            fontweight=font_cfg.get("weight", "bold"),
            **kwargs,
        )


    def set_xticks(self, ticks, labels=None, **kwargs):
        """设置X轴刻度"""
        plt.xticks(ticks, labels, **kwargs)

    def set_yticks(self, ticks, labels=None, **kwargs):
        """设置Y轴刻度"""
        plt.yticks(ticks, labels, **kwargs)

    def _apply_labels(self, xlabel=None, ylabel=None, title=None):
        """一次性应用xlabel, ylabel, title"""
        if xlabel: plt.xlabel(xlabel)
        if ylabel: plt.ylabel(ylabel)
        if title:  plt.title(title)

    def suptitle(self, text, **kwargs):
        """整张图的总标题"""
        font_cfg = self.config.get("font", {})
        plt.suptitle(
            text,
            fontsize=font_cfg.get("title_size", 14),
            fontweight=font_cfg.get("title_weight", "bold"),
            **kwargs,
        )

    def tight_layout(self, **kwargs):
        """自动调整子图间距"""
        plt.tight_layout(**kwargs)


    def savefig(self, path="output.png", dpi=300, **kwargs):
        plt.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)

    def show(self):
        """显示图像，兼容 Notebook 和 .py 脚本，避免重复显示"""
        if in_notebook():
            from IPython.display import display
            fig = plt.gcf()
            display(fig)
            plt.close(fig)   #  关键：关闭 fig，避免 Notebook 再自动渲染一次
        else:
            plt.show()


    def close(self):
        plt.close()
