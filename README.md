# alanmartyn.com

The source code for my personal website: [www.alanmartyn.com](https://www.alanmartyn.com)

A static website generator that renders content from Jupyter Notebooks and Markdown.

## To create a post

1. Create a post directory in `./input` to contain your post e.g. `./input/my-new-post`
1. Add your content file as markdown or jupyter notebook e.g. `./input/my-new-post/my-new-post.ipynb`
1. Add your post to the index `./index,json`
1. Render to static site `python render.py`
1. Git commit and push to publish via CircleCI continuous integration

#### Options

- If your post requires linked assets add an alias of `./input/assets/` 
- Set `"index": true` to display on homepage index, or `false` to publish page without link from index
- Homepage images should be 700x400px and live in `./input/assets`


## Design principles

- [x] should render jupyter notebooks
- [x] should render markdown
- [x] should be able to link people to a single dir containing all source code for a post
- [] should be deployed from declared environment e.g. docker 


