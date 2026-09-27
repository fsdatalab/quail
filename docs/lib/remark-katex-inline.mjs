/**
 * Renders math/inlineMath to KaTeX HTML during the remark phase,
 * outputting mdxJsxFlowElement nodes that MDX can compile.
 */
import katex from 'katex';
import { visit } from 'unist-util-visit';

export default function remarkKatexInline() {
  return (tree) => {
    visit(tree, ['math', 'inlineMath'], (node, index, parent) => {
      const displayMode = node.type === 'math';
      const html = katex.renderToString(node.value, {
        displayMode,
        throwOnError: false,
      });

      const jsxType = displayMode ? 'mdxJsxFlowElement' : 'mdxJsxTextElement';
      const tagName = displayMode ? 'div' : 'span';

      parent.children[index] = {
        type: jsxType,
        name: tagName,
        attributes: [
          {
            type: 'mdxJsxAttribute',
            name: 'dangerouslySetInnerHTML',
            value: {
              type: 'mdxJsxAttributeValueExpression',
              value: `{__html: ${JSON.stringify(html)}}`,
              data: {
                estree: {
                  type: 'Program',
                  sourceType: 'module',
                  body: [{
                    type: 'ExpressionStatement',
                    expression: {
                      type: 'ObjectExpression',
                      properties: [{
                        type: 'Property',
                        key: { type: 'Identifier', name: '__html' },
                        value: { type: 'Literal', value: html },
                        kind: 'init',
                        method: false,
                        shorthand: false,
                        computed: false,
                      }],
                    },
                  }],
                },
              },
            },
          },
        ],
        children: [],
        data: { _mdxExplicitJsx: true },
      };
    });
  };
}
