# --- Result Model ---
class AgentResult(BaseModel):
    response: str = Field(description="Response to the customer")
    target_agent: Optional[str] = Field(description="Agent to hand over to", default=None)

# --- Dependencies ---
@dataclass
class AgentDependencies:
    conversation_id: str
    user_id: str

# --- Agents ---
triage_agent = Agent(
    "openai:gpt-4o",
    deps_type=AgentDependencies,
    result_type=AgentResult,
    system_prompt="""You are a triage agent that directs customers to the right department.
    For technical issues (internet, service outages, connection problems), set target_agent to 'tech_support'.
    For billing issues (payments, charges, invoices), set target_agent to 'billing'.
    For general inquiries, handle them yourself without setting a target_agent.
    """
)

tech_support = Agent(
    "openai:gpt-4o",
    deps_type=AgentDependencies,
    result_type=AgentResult,
    system_prompt="""You are a technical support agent. Handle technical issues.
    If a billing question comes up, set target_agent to 'billing' in your response.
    """
)

billing_agent = Agent(
    "openai:gpt-4o",
    deps_type=AgentDependencies,
    result_type=AgentResult,
    system_prompt="""You are a billing support agent. Handle billing issues.
    If a technical question comes up, set target_agent to 'tech_support' in your response.
    """
)

# --- Router ---
class AgentRouter:
    def __init__(self):
        self.agents = {
            "triage": triage_agent,
            "tech_support": tech_support,
            "billing": billing_agent
        }

    async def handle_conversation(self, prompt: str, deps: AgentDependencies) -> str:
        current_agent = self.agents["triage"]
        
        while True:
            # Run current agent
            result = await current_agent.run(prompt, deps=deps)
            
            # Check for handover
            if result.data.target_agent:
                print(f"Handing over to {result.data.target_agent}")
                current_agent = self.agents[result.data.target_agent]
                continue
            
            # No handover needed, return response
            return result.data.response