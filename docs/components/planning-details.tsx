type Owner = 'datafusion' | 'quail';

const ALGORITHMS = {
  derive: `lower(plan):
  replace each AI call with an AI operator over the rows it reads
  move a join call below one input when the call reads only one input
  ask each AI pair once for an outer, semi, or anti join
  restore unmatched rows by row id
  reuse an answer only while every returned row keeps it unchanged

derive(plan):
  move each filter term to the operator producing its columns
  give each AI operator only the answer columns read above it
  pass each limit, DISTINCT, semi join, or count threshold down
  pass each CASE or Boolean guard to the operator producing its answer

offer(node, condition):
  if node produces a column read by condition:
    add condition to the node's filter
  else if node preserves condition:
    offer condition to the child
  else:
    keep a DataFusion filter above node

enough(node, required_rows):
  if node is an AI operator:
    set fetch to required_rows
  else if node preserves required_rows:
    pass required_rows to the child
  else:
    keep the DataFusion limit above node`,
  claim: `claim(plan):
  remove each unused AI operator

  for each run of AI operators and each AI join:
    read one named table directly
    let DataFusion compute every other input

    try to bind the run as one model region
    if the region cannot run:
      compute prompt arguments below the region
      split the top operator into another region
      move one filter or fetch above the region
      try the smaller region again

  return every answer and source column read above the region
  run DataFusion optimization again with each region as one operator`,
  region: `plan_region(region, statistics):
  bind each AI operator to its prompt, table, and output columns

  for each table:
    search every order of up to 10 AI operators
    rank larger sets greedily

    for each operator order:
      estimate the rows left after each filter
      estimate cost with and without resident document KV
      join consecutive operators with one document head into a pipeline

  for each AI join:
    choose the join order and anchor with dynamic programming

  for each candidate:
    choose prefix sharing, attention path, and KV retention

  return the candidate with the lowest estimated GPU time`,
  execute: `guard_region_filters(physical_plan):
  follow each dynamic filter from its join or sort to a model region
  map each filter column by name through the physical operators
  accept the filter only when every column reaches the right region column

run_pipeline(operators, rows):
  for each row, while fewer than fetch rows have passed:
    admit the document and write its KV

    for each AI operator:
      if guard is false or the document is NULL:
        write a NULL answer and continue

      append the question after the resident document KV

      after each answer token:
        let DataFusion evaluate the available part of the filter
        stop and drop the row when the filter becomes false
        stop the answer when the filter becomes true and no answer is returned

      rewind KV to the end of the document

    emit the row with each returned answer`,
} as const;

export function PlannerAlgorithm({ name }: { name: keyof typeof ALGORITHMS }) {
  return (
    <pre className="planner-algorithm not-prose">
      <code>{ALGORITHMS[name]}</code>
    </pre>
  );
}

function PlanNode({ owner, children }: { owner: Owner; children: React.ReactNode }) {
  return <div className={`planning-node ${owner}`}>{children}</div>;
}

function Arrow() {
  return <div className="planning-arrow" aria-hidden="true">↑</div>;
}

function Stack({ children }: { children: React.ReactNode[] }) {
  return (
    <div className="planning-stack">
      {children.map((child, index) => (
        <div key={index}>
          {index > 0 ? <Arrow /> : null}
          {child}
        </div>
      ))}
    </div>
  );
}

export function PlanStagesFigure() {
  return (
    <figure className="planning-detail not-prose">
      <div className="planning-stage-grid">
        <section>
          <h3>After parsing</h3>
          <Stack>{[
            <PlanNode owner="datafusion">Limit 10</PlanNode>,
            <PlanNode owner="datafusion">Filter year and AI.IF answer</PlanNode>,
            <PlanNode owner="quail">AI.CLASSIFY</PlanNode>,
            <PlanNode owner="quail">AI.IF</PlanNode>,
            <PlanNode owner="datafusion">Scan reviews</PlanNode>,
          ]}</Stack>
          <p>Both AI operators would read every review.</p>
        </section>

        <section>
          <h3>After optimization</h3>
          <Stack>{[
            <PlanNode owner="datafusion">Projection: id, sentiment</PlanNode>,
            <PlanNode owner="quail"><b>AI.CLASSIFY</b><small>projection: sentiment · fetch: 10</small></PlanNode>,
            <PlanNode owner="quail"><b>AI.IF</b><small>filter: answer is yes</small></PlanNode>,
            <PlanNode owner="datafusion"><b>Scan reviews</b><small>filter: year = 2024</small></PlanNode>,
          ]}</Stack>
          <p>DataFusion moves the year filter. Quail moves the answer test and limit.</p>
        </section>

        <section>
          <h3>After region claiming</h3>
          <Stack>{[
            <PlanNode owner="datafusion">Projection: id, sentiment</PlanNode>,
            <div className="planning-region" key="region">
              <b>Model region 1</b>
              <PlanNode owner="quail">AI.CLASSIFY</PlanNode>
              <Arrow />
              <PlanNode owner="quail">AI.IF</PlanNode>
            </div>,
            <PlanNode owner="datafusion"><b>Scan reviews</b><small>filter: year = 2024</small></PlanNode>,
          ]}</Stack>
          <p>The adjacent AI operators form one model region.</p>
        </section>

        <section>
          <h3>Physical plan</h3>
          <Stack>{[
            <PlanNode owner="datafusion">Projection: id, sentiment</PlanNode>,
            <div className="planning-region solid" key="pipeline">
              <b>ModelRegionExec 1</b>
              <div className="planning-pipeline">
                <PlanNode owner="quail"><b>Document</b><small>writes KV</small></PlanNode>
                <span>→</span>
                <PlanNode owner="quail"><b>AI.IF</b><small>drops row on no</small></PlanNode>
                <span>→</span>
                <PlanNode owner="quail"><b>AI.CLASSIFY</b><small>reads KV</small></PlanNode>
              </div>
              <small>Admission stops after 10 rows pass.</small>
            </div>,
            <PlanNode owner="datafusion"><b>Scan reviews</b><small>filter: year = 2024</small></PlanNode>,
          ]}</Stack>
          <p>AI.IF runs first because it can remove rows.</p>
        </section>
      </div>
      <figcaption>
        Each panel shows the example plan after one planning stage. Rows flow
        upward. Blue nodes belong to DataFusion, and purple nodes belong to
        Quail.
      </figcaption>
    </figure>
  );
}

const DERIVE_ROWS = [
  ['Before Limit', 'columns: id, sentiment · tests: none · rows needed: all', '—'],
  ['After Limit 10', 'columns: id, sentiment · tests: none · rows needed: 10', '—'],
  ['After Filter', 'columns: id, sentiment · test: AI.IF answer · rows needed: 10', '—'],
  ['After AI.CLASSIFY', 'columns: id, text · test: AI.IF answer · rows needed: 10', 'projection: sentiment · fetch: 10'],
  ['After AI.IF', 'columns: id, text · tests: none · rows needed: 10', 'filter: answer is yes'],
] as const;

export function DeriveFigure() {
  return (
    <figure className="planning-detail not-prose">
      <div className="planning-table-scroll">
        <table className="derive-table">
          <thead>
            <tr>
              <th>Plan position</th>
              <th>Requirements carried down</th>
              <th>Option written</th>
            </tr>
          </thead>
          <tbody>
            {DERIVE_ROWS.map(([position, note, option]) => (
              <tr key={position}>
                <td>{position}</td>
                <td>{note}</td>
                <td>{option}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <figcaption>
        <code>derive</code> walks from the result toward the scans. Plain
        nodes add requirements. Each AI operator copies the requirements it
        can apply into its options.
      </figcaption>
    </figure>
  );
}

export function SeveralRegionsFigure() {
  return (
    <figure className="planning-detail not-prose">
      <div className="planning-tree">
        <PlanNode owner="datafusion"><b>Aggregate</b><small>group by genre, topic · count</small></PlanNode>
        <Arrow />
        <PlanNode owner="datafusion"><b>Hash join</b><small>review.critic_id = critic.id</small></PlanNode>
        <div className="tree-children two">
          <div className="tree-branch">
            <Arrow />
            <PlanNode owner="datafusion"><b>Hash join</b><small>review.movie_id = movie.id</small></PlanNode>
            <div className="tree-children two">
              <div className="tree-branch">
                <Arrow />
                <div className="planning-region solid">
                  <b>Model region 1</b>
                  <div className="planning-pipeline compact">
                    <PlanNode owner="quail">AI.IF negative?</PlanNode>
                    <span>→</span>
                    <PlanNode owner="quail">AI.CLASSIFY topic</PlanNode>
                  </div>
                  <small>Dynamic filter: passing movie ids</small>
                </div>
                <Arrow />
                <PlanNode owner="datafusion">Scan reviews</PlanNode>
              </div>
              <div className="tree-branch">
                <Arrow />
                <div className="planning-region solid">
                  <b>Model region 2</b>
                  <PlanNode owner="quail">AI.IF time travel?</PlanNode>
                </div>
                <Arrow />
                <PlanNode owner="datafusion">Scan movies</PlanNode>
              </div>
            </div>
          </div>
          <div className="tree-branch narrow">
            <Arrow />
            <PlanNode owner="datafusion"><b>Scan critics</b><small>filter: country = US</small></PlanNode>
          </div>
        </div>
      </div>
      <figcaption>
        A plain hash join separates the AI operators over reviews from the AI
        operator over movies. DataFusion passes the ids of matching movies to
        model region 1 as a dynamic filter.
      </figcaption>
    </figure>
  );
}
