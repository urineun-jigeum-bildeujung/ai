"""분석에 사용하는 공개 데이터 파일의 출처와 무결성 정보를 정의합니다.

다운로드 코드와 로더가 같은 파일명·버전·라이선스·해시를 참조하도록
데이터셋 등록 정보를 한곳에서 관리합니다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetFile:
    """하나의 배포 파일을 식별하고 검증하는 데 필요한 고정 메타데이터입니다."""

    dataset: str
    original_publisher: str
    original_source_url: str
    distribution_name: str
    distribution_version: str
    distribution_page_url: str
    license_name: str
    license_url: str
    license_scope: str
    filename: str
    url: str
    expected_sha256: str
    archive_member: str | None = None


UCI_ONLINE_RETAIL_II = DatasetFile(
    dataset="uci_online_retail_ii",
    original_publisher="UCI Machine Learning Repository",
    original_source_url=("https://archive.ics.uci.edu/dataset/502/online+retail+ii"),
    distribution_name="UCI dataset 502",
    distribution_version="doi:10.24432/C5CG6D",
    distribution_page_url=("https://archive.ics.uci.edu/dataset/502/online+retail+ii"),
    license_name="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    license_scope="UCI Online Retail II 데이터셋 배포본",
    filename="online_retail_ii.zip",
    url=("https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"),
    expected_sha256=(
        "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"
    ),
    archive_member="online_retail_II.xlsx",
)

COMPLETE_JOURNEY_TRANSACTIONS = DatasetFile(
    dataset="complete_journey",
    original_publisher="84.51°",
    original_source_url="https://www.dunnhumby.com/source-files/",
    distribution_name="completejourney R package",
    distribution_version="1.1.1",
    distribution_page_url=("https://cran.r-project.org/package=completejourney"),
    license_name="CC0",
    license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    license_scope="completejourney 1.1.1 패키지 배포본",
    filename="transactions.rds",
    url=(
        "https://raw.githubusercontent.com/bradleyboehmke/"
        "completejourney/master/data/transactions.rds"
    ),
    expected_sha256=(
        "1fa0700033f1e5d9bb6b09e2be063d8d68474d346e95c50f2833e09d083e0007"
    ),
)

COMPLETE_JOURNEY_PRODUCTS = DatasetFile(
    dataset="complete_journey",
    original_publisher="84.51°",
    original_source_url="https://www.dunnhumby.com/source-files/",
    distribution_name="completejourney R package",
    distribution_version="1.1.1",
    distribution_page_url=("https://cran.r-project.org/package=completejourney"),
    license_name="CC0",
    license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    license_scope="completejourney 1.1.1 패키지 배포본",
    filename="products.rda",
    url=(
        "https://raw.githubusercontent.com/bradleyboehmke/"
        "completejourney/master/data/products.rda"
    ),
    expected_sha256=(
        "a80c6df33623b4af296ae9f317a6647e369db7e8ce7e7baed0e1bf44b9d979e5"
    ),
)

DATASET_FILES = (
    UCI_ONLINE_RETAIL_II,
    COMPLETE_JOURNEY_TRANSACTIONS,
    COMPLETE_JOURNEY_PRODUCTS,
)
