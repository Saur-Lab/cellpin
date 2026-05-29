{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% block methods %}
{% if methods %}
Methods table
~~~~~~~~~~~~~

.. autosummary::
{% for item in methods %}
    {%- if not item in inherited_members %}
    ~{{ name }}.{{ item }}
    {%- endif %}
{%- endfor %}
{% endif %}
{% endblock %}

{% block attributes_documentation %}
{% if attributes %}
Attributes
~~~~~~~~~~

{% for item in attributes %}
    {%- if not item in inherited_members %}
.. autoattribute:: {{ [objname, item] | join(".") }}
    {%- endif %}
{%- endfor %}

{% endif %}
{% endblock %}

{% block methods_documentation %}
{% if methods %}
Methods
~~~~~~~

{% for item in methods %}
    {%- if item != '__init__' and not item in inherited_members %}
.. automethod:: {{ [objname, item] | join(".") }}
    {%- endif -%}
{%- endfor %}

{% endif %}
{% endblock %}
