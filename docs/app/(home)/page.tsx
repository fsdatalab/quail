import Link from 'next/link';

export default function HomePage() {
  return (
    <div className="flex flex-col justify-center text-center flex-1 gap-4 px-6">
      <h1 className="text-3xl font-bold">Quail</h1>
      <p className="max-w-xl mx-auto text-fd-muted-foreground">
        A query engine for AI_FILTER and AI_JOIN over document collections.
        One model copy per H100, run through Modal.
      </p>
      <div className="flex gap-4 justify-center">
        <Link href="/docs" className="font-medium underline">
          Read the docs
        </Link>
        <Link href="/docs/user-guide/quickstart" className="font-medium underline">
          Quickstart
        </Link>
      </div>
    </div>
  );
}
