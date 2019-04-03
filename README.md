# alanmartyn.com

The source code for my personal website: [www.alanmartyn.com](https://www.alanmartyn.com) Includes a static website generator that renders content from Jupyter Notebooks and Markdown.


## Design principles

To articulate my work effectively requires a means for people to:
- [x] view my work in one place
- [x] view posts written as jupyter notebooks in the browser
- [x] view posts written markdown as in the browser
- [x] access source code for a post
- [x] execute source code for a post

## Building the site

Clone the repository

```
$ git clone https://github.com/a-martyn/website.git
```

Install dependencies (assumes Python 3 is installed)

```
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

Check settings in `config.py`.  
Run tests and build html

```
$ python test.py
$ python render.py
```

Preview site by opening html files `./output` in your browser.

## Creating a post

Make a directory to contain your post

```
$ mkdir input/my-new-post
```

Create content as markdown or jupyter notebook 

```
$ touch input/my-new-post/my-new-post.ipynb
```

Add your post to the index `./input/_index.json`  
Render to static site 

```
$ python render.py
````

Publish: `git commit` and `git push` to publish via CircleCI continuous integration


## Optional configuration

If your post requires linked images add `assets` dir and include linked images in there.

```
$ mkdir input/my-new-post/assets
```
Note: Asset filenames must be unique across all posts else `render.py` will raise an exception.  

To display your post on the homepage set `"index": true` in `input/_index.json`, or `false` to publish page without link from index.

Homepage images should be 700x400px and live in `./input/_index_assets`


