# alanmartyn.com

The source code for my personal website: [www.alanmartyn.com](https://www.alanmartyn.com) Includes a static website generator that renders content from Jupyter Notebooks and Markdown.


## Design principles

To articulate my work effectively requires a means to:
- [x] curate my work in one place
- [x] publish jupyter notebooks as html
- [x] publish markdown as html
- [x] link to a single dir containing all source code for a post

## To create a post

1. **Setup:** create a directory to contain your post e.g. `./input/content/my-new-post`
1. **Add content:** include content as markdown or jupyter notebook e.g. `./input/content/my-new-post/my-new-post.ipynb`
1. **Index:** add your post to the index `./index,json`
1. **Render:** to static site `python render.py`
1. **Publish:** `git commit` and `git push` to publish via CircleCI continuous integration


## Optional configuration

- If your post requires linked assets add symlink to `./input/content/assets/`, remember to use full path e.g. `ln -s full/path/to/input/content/assets/ full/path/to/input/content/my-post/assets`
- Set `"index": true` to display on homepage index, or `false` to publish page without link from index
- Homepage images should be 700x400px and live in `./input/content/assets`


