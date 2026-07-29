import { Fragment, useEffect, useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import type { UIMessage } from "@/types";
import { TextMessage } from "./messages/TextMessage";
import { AgentWorkBlock } from "./messages/AgentWorkBlock";
import { ChartMessage } from "./messages/ChartMessage";
import { groupIntoTurns, type TurnGroup } from "@/lib/groupMessages";

interface Props {
  messages: UIMessage[];
  isStreaming?: boolean;
  showReasoning?: boolean;
  onChartError?: (error: string) => void;
  /**
   * Render every turn in normal document flow instead of virtualizing.
   * Set by ChatView during PDF export so the whole transcript is in the DOM
   * and the `@media print` rules (which target `#chat-messages > [data-print-role]`)
   * still apply. Off for the interactive view, which virtualizes to stay bounded.
   */
  printing?: boolean;
}

/** The visible content of one turn. Shared by the virtualized and print paths. */
function TurnContent({
  group,
  showReasoning,
  onChartError,
}: {
  group: TurnGroup;
  showReasoning: boolean;
  onChartError?: (error: string) => void;
}) {
  return (
    <>
      {/* User message */}
      {group.userMsg && (
        <div className="flex flex-col items-end" data-print-role="user">
          <TextMessage payload={group.userMsg.payload as never} role="user" isStreaming={false} />
        </div>
      )}

      {/* Agent work block — tool calls, SQL, thinking. Charts are excluded. */}
      {group.workMsgs.length > 0 && (
        <AgentWorkBlock
          workMessages={group.workMsgs}
          turnUsage={group.finalMsg?.turnUsage}
          isStreaming={group.isActivelyStreaming}
          showReasoning={showReasoning}
          onChartError={onChartError}
        />
      )}

      {/* Charts rendered OUTSIDE the work block so they stay visible when it collapses. */}
      {group.chartMsgs.map((msg) => (
        <div key={msg.id} className="w-full" data-print-role="chart">
          <ChartMessage payload={msg.payload as never} onRenderError={onChartError} />
        </div>
      ))}

      {/* Final visible response */}
      {group.finalMsg && (group.finalMsg.payload as { text?: string }).text?.trim() && (
        <div className="flex flex-col items-start" data-print-role="assistant">
          <TextMessage
            payload={group.finalMsg.payload as never}
            role="assistant"
            isStreaming={group.finalMsg.isStreaming}
          />
        </div>
      )}
    </>
  );
}

export function MessageList({
  messages,
  isStreaming = false,
  showReasoning = true,
  onChartError,
  printing = false,
}: Props) {
  const parentRef = useRef<HTMLDivElement>(null);
  // Whether the user is pinned to the bottom. Starts true so a freshly opened
  // conversation lands on the latest turn; flipped off once they scroll up to
  // read history, so streaming/new turns don't yank them back down.
  const atBottomRef = useRef(true);

  const groups = messages.length > 0 ? groupIntoTurns(messages, isStreaming) : [];

  const virtualizer = useVirtualizer({
    count: groups.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 140,
    overscan: 6,
    getItemKey: (index) => groups[index].key,
  });

  // Keep the newest turn in view while the user is at the bottom — covers both
  // a new turn arriving and the final message growing token-by-token during
  // streaming (`messages` identity changes on each streamed chunk).
  useEffect(() => {
    if (printing || !atBottomRef.current || groups.length === 0) return;
    virtualizer.scrollToIndex(groups.length - 1, { align: "end" });
  }, [messages, groups.length, printing, virtualizer]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
        Ask a question about your data
      </div>
    );
  }

  // Print path: full transcript in normal flow, so the existing @media print
  // CSS (direct-child `data-print-role` selectors) works unchanged.
  if (printing) {
    return (
      // Fragment (not a wrapping div) keeps each turn's `data-print-role`
      // elements as *direct* children of #chat-messages, so the @media print
      // rules `#chat-messages > [data-print-role]` still match. The container's
      // own `space-y-3` supplies the inter-element gap, as in the pre-virtualized
      // version.
      <div id="chat-messages" className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        {groups.map((group) => (
          <Fragment key={group.key}>
            <TurnContent group={group} showReasoning={showReasoning} onChartError={onChartError} />
          </Fragment>
        ))}
      </div>
    );
  }

  const handleScroll = () => {
    const el = parentRef.current;
    if (!el) return;
    // 80px of slack: "near the bottom" still counts as pinned.
    atBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  };

  return (
    <div
      id="chat-messages"
      ref={parentRef}
      onScroll={handleScroll}
      className="flex-1 overflow-y-auto px-4 py-4"
    >
      <div style={{ height: virtualizer.getTotalSize(), position: "relative", width: "100%" }}>
        {virtualizer.getVirtualItems().map((item) => (
          <div
            key={item.key}
            data-index={item.index}
            ref={virtualizer.measureElement}
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              transform: `translateY(${item.start}px)`,
            }}
          >
            {/* pb-3 reproduces the old `space-y-3` gap; it lives inside the
                measured element so each row's height includes the gap. */}
            <div className="pb-3 space-y-3">
              <TurnContent
                group={groups[item.index]}
                showReasoning={showReasoning}
                onChartError={onChartError}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
