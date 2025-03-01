import { Marked } from "marked";

const markdown = new Marked({
  gfm: true,
  breaks: true,
});

markdown.Renderer.prototype.link = function ({ href, title, text }) {
  return `<a class="link" target="_blank" href="${href}" title="${title}">${text}</a>`;
};

export default markdown;
