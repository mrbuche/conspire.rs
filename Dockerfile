FROM python:latest
RUN curl https://sh.rustup.rs -fsSL | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
ARG VERSION
RUN if [ -z "$VERSION" ]; then \
        pip install git+https://github.com/mrbuche/conspire.py.git pytest; \
        git clone https://github.com/mrbuche/conspire.py.git; \
    else \
        pip install conspire==$VERSION pytest; \
        git clone https://github.com/mrbuche/conspire.py.git --branch $VERSION; \
    fi
RUN mkdir -p /usr/conspire/python/
RUN mv conspire.py/pyproject.toml /usr/conspire/python/
RUN mv conspire.py/tests/ /usr/conspire/python/
RUN rm -r conspire.py/
