FROM alpine:3.20

RUN mkdir -p /seed/workspace /workspace
COPY workspace_seed.tar /seed/workspace_seed.tar
RUN tar -xf /seed/workspace_seed.tar -C /seed/workspace \
    && rm /seed/workspace_seed.tar \
    && test -d /seed/workspace/runtime/mammography_metarepository/.git

CMD ["sh", "-ec", "if [ -e /workspace/.production-runtime-seeded ]; then echo 'Runtime workspace already seeded; preserving existing contents.'; exit 0; fi; if [ -n \"$(find /workspace -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)\" ]; then echo 'Refusing to seed a non-empty production runtime workspace without marker.' >&2; exit 20; fi; cp -a /seed/workspace/. /workspace/; printf '%s\\n' seeded > /workspace/.production-runtime-seeded; echo 'Runtime assets seeded successfully.'"]
