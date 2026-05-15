interface AgentAvatarProps {
  agentId: string
  label?: string
  size?: number
}

function KaiAvatar({ size }: { size: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" fill="none">
      <circle cx="20" cy="20" r="19" fill="#3a3a3c" stroke="#52525b" strokeWidth="1" />
      <circle cx="20" cy="22" r="11" fill="#e8d5b7" />
      <path d="M9 18 C9 10, 14 5, 20 4 C26 5, 31 10, 31 18" fill="#2c2c2e" />
      <path d="M12 15 L14 8 L17 14" fill="#2c2c2e" />
      <path d="M17 13 L20 5 L23 13" fill="#2c2c2e" />
      <path d="M23 14 L26 8 L28 15" fill="#2c2c2e" />
      <circle cx="16" cy="22" r="2" fill="#2c2c2e" />
      <circle cx="24" cy="22" r="2" fill="#2c2c2e" />
      <circle cx="16.6" cy="21.5" r="0.7" fill="white" />
      <circle cx="24.6" cy="21.5" r="0.7" fill="white" />
      <path d="M17 27 Q20 30 23 27" stroke="#2c2c2e" strokeWidth="1.2" fill="none" strokeLinecap="round" />
    </svg>
  )
}

function JohnAvatar({ size }: { size: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" fill="none">
      <circle cx="20" cy="20" r="19" fill="#3a3a3c" stroke="#52525b" strokeWidth="1" />
      <circle cx="20" cy="22" r="11" fill="#f0d9b5" />
      <path d="M9 19 C8 8, 15 4, 20 4 C25 4, 32 8, 31 19" fill="#5c3d2e" />
      <circle cx="11" cy="14" r="3" fill="#5c3d2e" />
      <circle cx="16" cy="10" r="3" fill="#5c3d2e" />
      <circle cx="21" cy="8" r="3" fill="#5c3d2e" />
      <circle cx="26" cy="10" r="3" fill="#5c3d2e" />
      <circle cx="29" cy="14" r="3" fill="#5c3d2e" />
      <circle cx="16" cy="22" r="3.5" stroke="#2c2c2e" strokeWidth="1.5" fill="none" />
      <circle cx="24" cy="22" r="3.5" stroke="#2c2c2e" strokeWidth="1.5" fill="none" />
      <line x1="19.5" y1="22" x2="20.5" y2="22" stroke="#2c2c2e" strokeWidth="1.2" />
      <line x1="9" y1="21" x2="12.5" y2="22" stroke="#2c2c2e" strokeWidth="1" />
      <line x1="27.5" y1="22" x2="31" y2="21" stroke="#2c2c2e" strokeWidth="1" />
      <circle cx="16" cy="22" r="1.5" fill="#2c2c2e" />
      <circle cx="24" cy="22" r="1.5" fill="#2c2c2e" />
      <circle cx="16.5" cy="21.5" r="0.5" fill="white" />
      <circle cx="24.5" cy="21.5" r="0.5" fill="white" />
      <line x1="18" y1="28" x2="22" y2="28" stroke="#2c2c2e" strokeWidth="1.2" strokeLinecap="round" />
    </svg>
  )
}

function RAvatar({ size }: { size: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" fill="none">
      <circle cx="20" cy="20" r="19" fill="#3a3a3c" stroke="#52525b" strokeWidth="1" />
      <circle cx="20" cy="22" r="11" fill="#d4a574" />
      <path d="M9 19 C9 9, 14 5, 20 5 C26 5, 31 9, 31 19 L31 16 C31 8, 26 4, 20 4 C14 4, 9 8, 9 16 Z" fill="#1a1a1a" />
      <ellipse cx="16" cy="21" rx="1.8" ry="2" fill="#2c2c2e" />
      <ellipse cx="24" cy="21" rx="1.8" ry="2" fill="#2c2c2e" />
      <circle cx="16.5" cy="20.5" r="0.6" fill="white" />
      <circle cx="24.5" cy="20.5" r="0.6" fill="white" />
      <line x1="14" y1="17.5" x2="18" y2="17" stroke="#1a1a1a" strokeWidth="1.3" strokeLinecap="round" />
      <line x1="22" y1="17" x2="26" y2="17.5" stroke="#1a1a1a" strokeWidth="1.3" strokeLinecap="round" />
      <path d="M18 29 Q20 32 22 29" fill="#1a1a1a" />
      <rect x="19" y="27" width="2" height="3" rx="1" fill="#1a1a1a" />
      <path d="M18 27 Q20 28.5 22 27" stroke="#2c2c2e" strokeWidth="0.8" fill="none" />
    </svg>
  )
}

const AVATAR_MAP: Record<string, React.FC<{ size: number }>> = {
  kai: KaiAvatar,
  john: JohnAvatar,
  r: RAvatar,
}

const FALLBACK_COLORS = [
  { bg: '#14532d', border: '#22c55e' },
  { bg: '#1d4ed8', border: '#60a5fa' },
  { bg: '#7c2d12', border: '#fb923c' },
  { bg: '#701a75', border: '#d946ef' },
  { bg: '#0f766e', border: '#2dd4bf' },
  { bg: '#9a3412', border: '#f97316' },
]

function hashString(value: string): number {
  return Array.from(value).reduce(
    (total, character) => total + character.charCodeAt(0),
    0,
  )
}

function buildInitials(agentId: string, label?: string): string {
  const source = (label || agentId).trim()
  if (!source) {
    return '?'
  }

  const words = source.split(/\s+/).filter(Boolean)
  if (words.length === 1) {
    return words[0].slice(0, 2).toUpperCase()
  }

  return `${words[0][0]}${words[1][0]}`.toUpperCase()
}

export default function AgentAvatar({
  agentId,
  label,
  size = 36,
}: AgentAvatarProps) {
  const Avatar = AVATAR_MAP[agentId]
  if (Avatar) {
    return <Avatar size={size} />
  }

  const palette = FALLBACK_COLORS[hashString(agentId) % FALLBACK_COLORS.length]

  return (
    <div
      className="rounded-full flex items-center justify-center text-white font-bold text-xs uppercase"
      style={{
        width: size,
        height: size,
        backgroundColor: palette.bg,
        border: `1px solid ${palette.border}`,
      }}
    >
      {buildInitials(agentId, label)}
    </div>
  )
}
