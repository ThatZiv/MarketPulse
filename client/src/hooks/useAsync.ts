// taken from https://github.com/sergeyleschev/react-custom-hooks/blob/main/src/hooks/useAsync/useAsync.js

import { useCallback, useEffect, useState } from "react";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export default function useAsync<T = any>(
  callback: () => Promise<T>,
  dependencies: unknown[] = []
) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>();
  const [value, setValue] = useState<T>();

  const callbackMemoized = useCallback(() => {
    setLoading(true);
    setError(undefined);
    setValue(undefined);
    callback()
      .then(setValue)
      .catch(setError)
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, dependencies);

  useEffect(() => {
    callbackMemoized();
  }, [callbackMemoized]);

  return { loading, error, value };
}
