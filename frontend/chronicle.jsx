const { useState, useEffect, useRef } = React;

      const LucideIcon = ({ name, className }) => {
        const icon = window.lucide?.icons?.[name];
        if (!icon) return null;
        const renderNode = (node, key) => {
          if (!node) return null;
          const [tag, attrs, children] = node;
          return React.createElement(
            tag,
            { ...attrs, key, stroke: 'currentColor', fill: 'none' },
            children?.map((child, index) => renderNode(child, `${key}-${index}`))
          );
        };
        const [tag, attrs, children] = icon;
        return React.createElement(
          tag,
          { ...attrs, className, stroke: 'currentColor', fill: 'none' },
          children?.map((child, index) => renderNode(child, index))
        );
      };

      const formatTimestamp = (timestamp) => {
        if (!timestamp) return 'Unknown';
        try {
          return new Date(timestamp).toLocaleString();
        } catch (err) {
          return timestamp;
        }
      };

      const buildMutationEvents = (edges = [], nodes = []) => {
        const nodeMap = nodes.reduce((acc, node) => {
          acc[node.id] = node;
          return acc;
        }, {});
        return edges
          .filter((edge) => edge.attributes?.relation_type === 'Mutation')
          .sort((a, b) => (b.attributes?.mutation_score ?? 0) - (a.attributes?.mutation_score ?? 0))
          .slice(0, 5)
          .map((edge) => ({
            node: edge.target,
            score: edge.attributes?.mutation_score ?? 0,
            type: 'mutation',
            timestamp: nodeMap[edge.target]?.timestamp || 'N/A',
            title: nodeMap[edge.target]?.content?.slice(0, 80) || edge.target,
          }));
      };

      const enrichNodes = (graph) => {
        const edges = graph.edges || [];
        const scoreMap = edges.reduce((acc, edge) => {
          if (edge.target) {
            acc[edge.target] = edge.attributes?.mutation_score ?? 0;
          }
          return acc;
        }, {});
        return (graph.nodes || []).map((node) => ({
          id: node.id,
          title: node.content?.split('. ')[0] || node.id,
          url: node.id,
          mutationScore: scoreMap[node.id] ?? 0,
          timestamp: node.timestamp,
          author: node.author || 'Unknown',
          depth: node.depth || 0,
          content: node.content || '',
        }));
      };

      function PhylosUI() {
        const [url, setUrl] = useState('');
        const [maxDepth, setMaxDepth] = useState(3);
        const [isAnalyzing, setIsAnalyzing] = useState(false);
        const [graphData, setGraphData] = useState(null);
        const [selectedNode, setSelectedNode] = useState(null);
        const [mutationEvents, setMutationEvents] = useState([]);
        const [telemetry, setTelemetry] = useState({ events: 0, nodes: 0, mutations: 0, replications: 0 });
        const [summary, setSummary] = useState('Submit a URL to begin a trace.');
        const [sessionId, setSessionId] = useState(null);
        const [chatMessages, setChatMessages] = useState([
          { role: 'system', text: 'Submit a source URL and select "Trace Origin" to begin.' },
        ]);
        const [chatInput, setChatInput] = useState('');
        const wsRef = useRef(null);

        useEffect(() => {
          return () => {
            if (wsRef.current) {
              wsRef.current.close();
            }
          };
        }, []);

        const appendMessage = (role, text) => {
          setChatMessages((prev) => [...prev, { role, text }]);
        };

        const fetchGraphSnapshot = async (id) => {
          try {
            const response = await fetch(`/session/${id}/graph`);
            if (!response.ok) {
              throw new Error('Failed to load graph snapshot');
            }
            const payload = await response.json();
            const enrichedNodes = enrichNodes(payload);
            setGraphData({ nodes: enrichedNodes, edges: payload.edges || [] });
            if (enrichedNodes.length) {
              setSelectedNode(enrichedNodes[0]);
            }
            setMutationEvents(buildMutationEvents(payload.edges, enrichedNodes));
            setTelemetry((prev) => ({
              ...prev,
              nodes: payload.stats?.nodes ?? prev.nodes,
              mutations: payload.stats?.mutations ?? prev.mutations,
              replications: payload.stats?.replications ?? prev.replications,
            }));
          } catch (error) {
            console.error(error);
            appendMessage('system', 'Unable to load final graph snapshot.');
          }
        };

        const handleSocketMessage = (message) => {
          if (message.status) {
            if (message.status === 'session') {
              setSessionId(message.session_id);
              appendMessage('system', `Session established (${message.session_id.slice(0, 8)}…).`);
              return;
            }
            if (message.status === 'summary') {
              setSummary(message.summary || 'Summary unavailable.');
              appendMessage('assistant', 'Trace complete. Ask me what you would like to investigate.');
              setIsAnalyzing(false);
              if (message.session_id) {
                fetchGraphSnapshot(message.session_id);
              } else if (sessionId) {
                fetchGraphSnapshot(sessionId);
              }
              return;
            }
            if (message.status === 'info') {
              appendMessage('system', message.message || 'Info update received.');
              return;
            }
            if (message.status === 'error') {
              appendMessage('system', message.message || 'An error occurred.');
              setIsAnalyzing(false);
              return;
            }
          }

          if (message.event) {
            setTelemetry((prev) => ({
              events: prev.events + 1,
              nodes: Math.max(prev.nodes, message.data?.knowledge_graph?.node_count || prev.nodes),
              mutations: prev.mutations + (message.data?.edge_stats?.mutation_count || 0),
              replications: prev.replications + (message.data?.edge_stats?.replication_count || 0),
            }));
          }
        };

        const startAnalysis = () => {
          if (!url || isAnalyzing) return;
          setIsAnalyzing(true);
          setGraphData(null);
          setSelectedNode(null);
          setMutationEvents([]);
          setTelemetry({ events: 0, nodes: 0, mutations: 0, replications: 0 });
          setSummary('Tracing narrative lineage...');
          setChatMessages([{ role: 'system', text: 'Investigation running… chat will unlock once complete.' }]);
          if (wsRef.current) {
            wsRef.current.close();
          }
          const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
          const wsUrl = `${protocol}://${window.location.host}/ws/dna-stream`;
          const ws = new WebSocket(wsUrl);
          wsRef.current = ws;

          ws.onopen = () => {
            ws.send(JSON.stringify({ start_url: url, max_depth: Number(maxDepth) || 1 }));
          };
          ws.onmessage = (event) => {
            try {
              const payload = JSON.parse(event.data);
              handleSocketMessage(payload);
            } catch (error) {
              console.error('Failed to parse WebSocket message', error);
            }
          };
          ws.onerror = () => {
            appendMessage('system', 'WebSocket error encountered.');
            setIsAnalyzing(false);
          };
          ws.onclose = () => {
            setIsAnalyzing(false);
          };
        };

        const sendChatMessage = async (evt) => {
          evt.preventDefault();
          if (!chatInput.trim() || !sessionId) {
            return;
          }
          const message = chatInput.trim();
          setChatInput('');
          appendMessage('user', message);
          try {
            const response = await fetch('/chat', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ session_id: sessionId, message }),
            });
            const payload = await response.json();
            if (!response.ok) {
              appendMessage('system', payload.detail || 'Chat failed.');
              return;
            }
            appendMessage('assistant', payload.reply || 'I need more context.');
          } catch (error) {
            appendMessage('system', 'Unable to reach chat endpoint.');
          }
        };

        const getMutationColor = (score) => {
          if (score < 0.2) return 'bg-emerald-800';
          if (score < 0.5) return 'bg-amber-700';
          return 'bg-red-900';
        };

        const getMutationLabel = (score) => {
          if (score < 0.2) return 'Authentic';
          if (score < 0.5) return 'Modified';
          return 'Corrupted';
        };

        const averageMutation = graphData?.nodes?.length
          ? graphData.nodes.reduce((acc, node) => acc + (node.mutationScore || 0), 0) / graphData.nodes.length
          : 0;

        return (
          <div className="relative">
            <div
              className="fixed inset-0 opacity-10 pointer-events-none"
              style={{
                backgroundImage:
                  "url('data:image/svg+xml,%3Csvg width=\\'100\\' height=\\'100\\' xmlns=\\'http://www.w3.org/2000/svg\\'%3E%3Cfilter id=\\'noise\\'%3E%3CfeTurbulence type=\\'fractalNoise\\' baseFrequency=\\'0.9\\' numOctaves=\\'4\\' /%3E%3C/filter%3E%3Crect width=\\'100\\' height=\\'100\\' filter=\\'url(%23noise)\\' opacity=\\'0.4\\'/%3E%3C/svg%3E')",
              }}
            />

            <div className="border-b-4 border-amber-800/30 bg-black/20 backdrop-blur-md relative overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-amber-900/5 to-transparent" />
              <div className="max-w-7xl mx-auto px-6 py-6 relative">
                <div className="text-center border-b border-amber-800/20 pb-4 mb-4">
                  <div className="flex items-center justify-center gap-3 mb-2">
                    <div className="h-px w-20 bg-gradient-to-r from-transparent to-amber-700/50" />
                    <LucideIcon name="Newspaper" className="w-5 h-5 text-amber-700/70" />
                    <div className="h-px w-20 bg-gradient-to-l from-transparent to-amber-700/50" />
                  </div>
                  <h1
                    className="text-6xl mb-2 text-amber-100/90 tracking-wide"
                    style={{
                      fontFamily: '"UnifrakturMaguntia", "Old English Text MT", "Blackletter", serif',
                      textShadow: '2px 2px 4px rgba(0,0,0,0.7)',
                      fontWeight: 400,
                    }}
                  >
                    Phylos Chronicle
                  </h1>
                  <div className="flex items-center justify-center gap-2 text-xs text-amber-700/70 uppercase tracking-widest">
                    <div className="h-px w-12 bg-amber-800/30" />
                    <span>Narrative Evolution Ledger</span>
                    <div className="h-px w-12 bg-amber-800/30" />
                  </div>
                </div>
                <p className="text-center text-sm text-amber-200/60 italic font-serif">
                  "Tracing the lineage of truth through the shadows of disinformation"
                </p>
              </div>
            </div>

            <div className="max-w-7xl mx-auto px-6 py-8 relative">
              <div className="mb-8">
                <div className="bg-gradient-to-br from-slate-800/40 to-slate-900/50 rounded-sm p-8 border-4 border-double border-amber-900/40 backdrop-blur-md shadow-2xl relative">
                  <div className="absolute top-4 left-4 text-6xl text-amber-900/10 font-serif">"</div>
                  <div className="absolute bottom-4 right-4 text-6xl text-amber-900/10 font-serif">"</div>

                  <div className="flex items-center gap-3 mb-4">
                    <LucideIcon name="Eye" className="w-5 h-5 text-amber-700" />
                    <h2 className="text-xl font-serif text-amber-100/90">Investigation Desk</h2>
                  </div>
                  <p className="text-amber-200/60 mb-6 text-sm font-serif italic">
                    Submit a source URL to trace its narrative mutations across the information network.
                  </p>

                  <div className="flex flex-col lg:flex-row gap-3">
                    <div className="flex-1 relative">
                      <LucideIcon name="Search" className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-amber-700/50" />
                      <input
                        type="text"
                        value={url}
                        onChange={(e) => setUrl(e.target.value)}
                        placeholder="https://source-article.com/original-story"
                        className="w-full bg-black/60 border-2 border-amber-900/40 rounded-sm px-12 py-4 text-amber-100 placeholder-amber-700/40 focus:outline-none focus:border-amber-700/60 focus:ring-2 focus:ring-amber-700/20 font-mono text-sm"
                        onKeyPress={(e) => e.key === 'Enter' && startAnalysis()}
                      />
                    </div>
                    <input
                      type="number"
                      min={1}
                      max={6}
                      value={maxDepth}
                      onChange={(e) => setMaxDepth(e.target.value)}
                      className="w-24 bg-black/60 border-2 border-amber-900/40 rounded-sm px-4 py-2 text-amber-100 font-mono text-sm"
                      title="Max Depth"
                    />
                    <button
                      onClick={startAnalysis}
                      disabled={isAnalyzing || !url}
                      className="px-8 py-4 bg-gradient-to-b from-amber-800 to-amber-900 rounded-sm font-serif uppercase text-sm tracking-wider hover:from-amber-700 hover:to-amber-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2 shadow-lg border border-amber-700/50"
                    >
                      {isAnalyzing ? (
                        <>
                          <div className="w-5 h-5 border-2 border-amber-200/30 border-t-amber-200 rounded-full animate-spin" />
                          Investigating
                        </>
                      ) : (
                        <>
                          <LucideIcon name="Zap" className="w-5 h-5" />
                          Trace Origin
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>

              {graphData ? (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <div className="lg:col-span-2 bg-gradient-to-br from-slate-900/50 to-slate-800/40 rounded-sm p-6 border-2 border-amber-900/30 backdrop-blur-md shadow-2xl">
                    <div className="border-b-2 border-amber-900/30 pb-4 mb-6">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <LucideIcon name="Link2" className="w-5 h-5 text-amber-700" />
                          <h3 className="text-xl font-serif text-amber-100/90">Citation Network</h3>
                        </div>
                        <div className="flex gap-3 text-xs font-serif">
                          <div className="flex items-center gap-1.5">
                            <div className="w-3 h-3 bg-emerald-800 border border-emerald-600" />
                            <span className="text-emerald-300">Authentic</span>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <div className="w-3 h-3 bg-amber-700 border border-amber-500" />
                            <span className="text-amber-300">Modified</span>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <div className="w-3 h-3 bg-red-900 border border-red-700" />
                            <span className="text-red-300">Corrupted</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-4">
                      {graphData.nodes.map((node) => (
                        <div
                          key={node.id}
                          onClick={() => setSelectedNode(node)}
                          className={`p-5 border-2 cursor-pointer transition-all ${
                            selectedNode?.id === node.id
                              ? 'bg-amber-900/20 border-amber-700/60 shadow-lg'
                              : 'bg-black/30 border-amber-900/30 hover:border-amber-800/50 hover:bg-amber-950/20'
                          }`}
                        >
                          <div className="flex items-start gap-4">
                            <div
                              className={`w-14 h-14 ${getMutationColor(node.mutationScore)} border-2 ${
                                node.mutationScore < 0.2
                                  ? 'border-emerald-600'
                                  : node.mutationScore < 0.5
                                  ? 'border-amber-500'
                                  : 'border-red-700'
                              } flex items-center justify-center flex-shrink-0 shadow-lg`}
                            >
                              {node.mutationScore < 0.2 ? (
                                <LucideIcon name="Shield" className="w-7 h-7 text-emerald-100" />
                              ) : node.mutationScore < 0.5 ? (
                                <LucideIcon name="TrendingUp" className="w-7 h-7 text-amber-100" />
                              ) : (
                                <LucideIcon name="AlertTriangle" className="w-7 h-7 text-red-100" />
                              )}
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-start justify-between mb-2">
                                <div className="flex-1">
                                  <h4 className="font-serif text-lg font-bold text-amber-50 mb-1">{node.title}</h4>
                                  <p className="text-xs text-amber-700/70 uppercase tracking-wide font-serif">By {node.author}</p>
                                </div>
                                <span
                                  className={`text-xs px-3 py-1 border-2 ml-4 ${
                                    node.mutationScore < 0.2
                                      ? 'bg-emerald-900/40 border-emerald-700 text-emerald-200'
                                      : node.mutationScore < 0.5
                                      ? 'bg-amber-900/40 border-amber-700 text-amber-200'
                                      : 'bg-red-900/40 border-red-700 text-red-200'
                                  } font-serif uppercase tracking-wider`}
                                >
                                  {getMutationLabel(node.mutationScore)}
                                </span>
                              </div>
                              <p className="text-sm text-amber-300/50 font-mono truncate border-l-2 border-amber-900/30 pl-3">{node.url}</p>
                              <div className="flex items-center gap-4 mt-3 text-xs text-amber-600/60 font-serif italic border-t border-amber-900/20 pt-2">
                                <span>Mutation Index: {node.mutationScore.toFixed(3)}</span>
                                <span>•</span>
                                <span>Published: {formatTimestamp(node.timestamp)}</span>
                              </div>
                            </div>
                            <LucideIcon name="ChevronRight" className="w-5 h-5 text-amber-700/40" />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-6">
                    <div className="bg-gradient-to-br from-slate-900/50 to-slate-800/40 rounded-sm p-6 border-2 border-amber-900/30 backdrop-blur-md shadow-2xl">
                      <h3 className="text-lg font-serif text-amber-100/90 mb-4 pb-2 border-b-2 border-amber-900/30 flex items-center gap-2">
                        <LucideIcon name="Activity" className="w-5 h-5 text-amber-700" />
                        Investigation Report
                      </h3>
                      <div className="space-y-4 font-serif">
                        <div className="flex justify-between items-center border-b border-amber-900/20 pb-2">
                          <span className="text-amber-300/70 text-sm">Events Streamed</span>
                          <span className="font-bold text-2xl text-amber-100">{telemetry.events}</span>
                        </div>
                        <div className="flex justify-between items-center border-b border-amber-900/20 pb-2">
                          <span className="text-amber-300/70 text-sm">Sources Traced</span>
                          <span className="font-bold text-2xl text-amber-100">{graphData.nodes.length}</span>
                        </div>
                        <div className="flex justify-between items-center border-b border-amber-900/20 pb-2">
                          <span className="text-amber-300/70 text-sm">Corruptions Found</span>
                          <span className="font-bold text-2xl text-red-300">{telemetry.mutations}</span>
                        </div>
                        <div className="flex justify-between items-center border-b border-amber-900/20 pb-2">
                          <span className="text-amber-300/70 text-sm">Replications Logged</span>
                          <span className="font-bold text-2xl text-amber-100">{telemetry.replications}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-amber-300/70 text-sm">Avg. Mutation</span>
                          <span className="font-bold text-2xl text-amber-100">{averageMutation.toFixed(3)}</span>
                        </div>
                      </div>
                    </div>

                    <div className="bg-gradient-to-br from-red-950/40 to-slate-900/50 rounded-sm p-6 border-2 border-red-900/40 backdrop-blur-md shadow-2xl">
                      <h3 className="text-lg font-serif text-red-200 mb-4 pb-2 border-b-2 border-red-900/30 flex items-center gap-2">
                        <LucideIcon name="AlertTriangle" className="w-5 h-5 text-red-400" />
                        Corruption Alerts
                      </h3>
                      <div className="space-y-3">
                        {mutationEvents.length ? (
                          mutationEvents.map((event, idx) => (
                            <div key={idx} className="p-4 bg-black/40 border-2 border-red-900/30">
                              <div className="flex items-center justify-between mb-2 pb-2 border-b border-red-900/20">
                                <span className="text-xs font-serif text-red-300 uppercase tracking-wider">
                                  {event.type.replace('_', ' ')}
                                </span>
                                <span className="text-sm text-red-200 font-bold font-mono">
                                  {event.score.toFixed(3)}
                                </span>
                              </div>
                              <p className="text-xs text-amber-400/60 font-serif italic">{formatTimestamp(event.timestamp)}</p>
                              <p className="text-sm text-amber-200/70 font-serif mt-2">{event.title}</p>
                            </div>
                          ))
                        ) : (
                          <p className="text-sm text-red-300/50 font-serif italic text-center py-4">
                            No major corruptions detected yet.
                          </p>
                        )}
                      </div>
                    </div>

                    {selectedNode && (
                      <div className="bg-gradient-to-br from-slate-900/50 to-slate-800/40 rounded-sm p-6 border-2 border-amber-900/30 backdrop-blur-md shadow-2xl">
                        <h3 className="text-lg font-serif text-amber-100/90 mb-4 pb-2 border-b-2 border-amber-900/30">
                          Article Details
                        </h3>
                        <div className="space-y-4 text-sm font-serif">
                          <div>
                            <span className="text-amber-600/80 block mb-1 uppercase text-xs tracking-wide">Headline</span>
                            <span className="font-medium text-amber-100">{selectedNode.title}</span>
                          </div>
                          <div>
                            <span className="text-amber-600/80 block mb-1 uppercase text-xs tracking-wide">Source URL</span>
                            <span className="font-mono text-xs text-amber-300/80 break-all">{selectedNode.url}</span>
                          </div>
                          <div>
                            <span className="text-amber-600/80 block mb-1 uppercase text-xs tracking-wide">Author</span>
                            <span className="text-amber-200/90">{selectedNode.author}</span>
                          </div>
                          <div>
                            <span className="text-amber-600/80 block mb-2 uppercase text-xs tracking-wide">Mutation Index</span>
                            <div className="flex items-center gap-3">
                              <div className="flex-1 h-3 bg-black/40 border border-amber-900/30 overflow-hidden">
                                <div
                                  className={`h-full ${getMutationColor(selectedNode.mutationScore)} border-r-2 ${
                                    selectedNode.mutationScore < 0.2
                                      ? 'border-emerald-500'
                                      : selectedNode.mutationScore < 0.5
                                      ? 'border-amber-500'
                                      : 'border-red-500'
                                  }`}
                                  style={{ width: `${Math.min(selectedNode.mutationScore, 1) * 100}%` }}
                                />
                              </div>
                              <span className="font-bold font-mono text-amber-100">{selectedNode.mutationScore.toFixed(3)}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                !isAnalyzing && (
                  <div className="text-center py-20">
                    <div className="w-32 h-32 bg-amber-950/20 border-4 border-double border-amber-900/40 flex items-center justify-center mx-auto mb-6">
                      <LucideIcon name="Newspaper" className="w-16 h-16 text-amber-700/40" />
                    </div>
                    <h3 className="text-2xl font-serif text-amber-200/70 mb-3">Awaiting Investigation</h3>
                    <p className="text-amber-400/50 text-sm font-serif italic max-w-md mx-auto">
                      Submit a source article above to begin tracing its narrative evolution through the shadows of the information age.
                    </p>
                  </div>
                )
              )}

              <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-gradient-to-br from-slate-900/50 to-slate-800/40 rounded-sm p-6 border-2 border-amber-900/30 backdrop-blur-md shadow-2xl">
                  <h3 className="text-lg font-serif text-amber-100/90 mb-3 flex items-center gap-2">
                    <LucideIcon name="Globe" className="w-5 h-5 text-amber-700" />
                    Gemini Debrief
                  </h3>
                  <div className="summary-box text-sm font-serif text-amber-200/80">{summary}</div>
                </div>
                <div className="bg-gradient-to-br from-slate-900/50 to-slate-800/40 rounded-sm p-6 border-2 border-amber-900/30 backdrop-blur-md shadow-2xl">
                  <h3 className="text-lg font-serif text-amber-100/90 mb-3 flex items-center gap-2">
                    <LucideIcon name="Activity" className="w-5 h-5 text-amber-700" />
                    Gemini Chat
                  </h3>
                  <div className="chat-panel">
                    <div id="chat-messages" className="chat-messages">
                      {chatMessages.map((entry, idx) => (
                        <div key={idx} className={`chat-entry ${entry.role}`}>
                          <div className="avatar">{entry.role === 'assistant' ? 'G' : entry.role === 'user' ? 'You' : '!'}</div>
                          <div className="bubble">
                            <strong className="font-serif text-sm capitalize">{entry.role}</strong>
                            <p className="font-serif text-sm text-amber-100/90">{entry.text}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                    <form id="chat-form" onSubmit={sendChatMessage} className="space-y-2">
                      <textarea
                        id="chat-input"
                        value={chatInput}
                        onChange={(e) => setChatInput(e.target.value)}
                        placeholder={sessionId ? 'Ask Gemini about the findings...' : 'Chat unlocks once a trace completes.'}
                        disabled={!sessionId}
                        className="w-full bg-black/60 border-2 border-amber-900/40 rounded-sm px-4 py-3 text-amber-100 placeholder-amber-700/40 focus:outline-none focus:border-amber-700/60 focus:ring-2 focus:ring-amber-700/20 font-serif text-sm"
                      ></textarea>
                      <button
                        id="chat-send"
                        type="submit"
                        disabled={!sessionId}
                        className="px-6 py-2 bg-gradient-to-b from-amber-800 to-amber-900 rounded-sm font-serif uppercase text-xs tracking-wider hover:from-amber-700 hover:to-amber-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2 shadow-lg border border-amber-700/50"
                      >
                        <LucideIcon name="MessageCircle" className="w-4 h-4" />
                        Ask Gemini
                      </button>
                    </form>
                  </div>
                </div>
              </div>
            </div>

            <div className="border-t-2 border-amber-900/30 bg-black/20 mt-12">
              <div className="max-w-7xl mx-auto px-6 py-4 text-center">
                <p className="text-xs text-amber-700/60 font-serif italic">Est. 2025 • Phylos Chronicle • Shedding Light on Digital Shadows</p>
              </div>
            </div>
          </div>
        );
      }

      ReactDOM.createRoot(document.getElementById('root')).render(<PhylosUI />);
