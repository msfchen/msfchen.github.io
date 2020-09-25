---
layout: default
---

<h1 class="post_title">
  <b>{{ page.title }}</b>
</h1>
  {% if page.date %}
  {%- assign date_format = site.minima.date_format | default: "%b %-d, %Y" -%}
  <h4 class="post_title">{{ page.date | date: date_format }} by Shuo-Fu Michael Chen</h4>
  {% endif %}


{{ content }}

{% include disqus.html %}